import numpy as np
import torch
from torch import tensor, float32
from Src.Algorithms.Agent import Agent
from Src.Utils import Basis, utils
from Src.Algorithms import NS_utils
from Src.Algorithms.Extrapolator import OLS

"""

"""
class ProOLS(Agent):
    def __init__(self, config):
        super(ProOLS, self).__init__(config)
        # Get state features and instances for Actor and Value function
        self.state_features = Basis.get_Basis(config=config)
        self.actor, self.atype, self.action_size = NS_utils.get_Policy(state_dim=self.state_features.feature_dim,
                                                                       config=config)
        self.memory = utils.TrajectoryBuffer(buffer_size=config.buffer_size, state_dim=self.state_dim,
                                             action_dim=self.action_size, atype=self.atype, config=config, dist_dim=1)
        self.extrapolator = OLS(max_len=config.buffer_size, delta=config.delta, basis_type=config.extrapolator_basis,
                                k=config.fourier_k)



        self.modules = [('actor', self.actor), ('state_features', self.state_features)]
        self.counter = 0
        self.init()

    def reset(self):
        super(ProOLS, self).reset()
        self.memory.next()
        self.counter += 1
        self.gamma_t = 1

    def get_action(self, state):
        state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device)
        state = self.state_features.forward(state.view(1, -1))
        action, prob, dist = self.actor.get_action_w_prob_dist(state)
        # if self.config.debug:
        #     self.track_entropy(dist, action)

        return action, prob, dist

    def update(self, s1, a1, prob, r1, s2, done):
        # Batch episode history
        self.memory.add(s1, a1, prob, self.gamma_t * r1)
        self.gamma_t *= self.config.gamma

        if done and self.counter % self.config.delta == 0:
            self.optimize()
        ## how does update function works ##
        # we optimize for every self.config.delta=5)
        # if self.optimize has been visited for N times, them we have N * self.config.delta buffer size.
        #
    def optimize(self):
        if self.memory.size <= self.config.fourier_k:
            # If number of rows is less than number of features (columns), it wont have full column rank.
            return

        batch_size = self.memory.size if self.memory.size < self.config.batch_size else self.config.batch_size
        # batch size increases
        # Compute and cache the partial derivatives w.r.t to each of the episodes
        self.extrapolator.update(self.memory.size, self.config.delta)

        # Inner optimization loop
        # Note: Works best with large number of iterations with small step-sizes
        for iter in range(self.config.max_inner):

            ################################################
            ### Algorithm1 step3 : compute PDIS gradient ###
            ###############################################
            id, s, a, beta, r, mask = self.memory.sample(batch_size)            # B, BxHxD, BxHxA, BxH, BxH, BxH
            # B : episode number
            # H : step number per episode
            # D : dimenstion
            B, H, D = s.shape
            _, _, A = a.shape

            # create state features
            s_feature = self.state_features.forward(s.view(B * H, D))           # BxHxD -> (BxH)xd

            # Get action probabilities
            log_pi, dist_all = self.actor.get_logprob_dist(s_feature, a.view(B * H, -1))     # (BxH)xd, (BxH)xA
            log_pi = log_pi.view(B, H)                                                       # (BxH)x1 -> BxH
            pi_a = torch.exp(log_pi)                                                         # (BxH)x1 -> BxH

            # Get importance ratios and log probabilities
            rho = (pi_a / beta).detach()                                        # BxH / BxH -> BxH

            # save pi_a for each timestep



            # Forward multiply all the rho to get probability of trajectory
            for i in range(1, H):
                rho[:, i] *= rho[:, i-1]

            rho = torch.clamp(rho, 0, self.config.importance_clip)              # Clipped Importance sampling (Biased)
            rho = rho * mask                                                    # BxH * BxH -> BxH

            # Create importance sampled rewards
            returns = rho * r                                                   # BxH * BxH -> BxH

            # Reverse sum all the returns to get actual returns
            for i in range(H-2, -1, -1):
                returns[:, i] += returns[:, i+1]

            loss = 0
            ########################################
            ### log_pi_return : equation (5)-(b) ###
            ########################################
            log_pi_return = torch.sum(log_pi * returns, dim=-1, keepdim=True)   # sum(BxH * BxH) -> Bx1

            # Get the Extrapolator gradients w.r.t Off-policy terms
            # Using the formula for the full derivative, we can compute this first part directly
            # to save compute time.
            ####################################################################################################
            ### del_extrapolator: equation (5)-(a)                                                           ###
            ### note : the function " derivatives" already considers forcasted future J_{k+1},..,J_{k+delta}  ##
            ####################################################################################################
            del_extrapolator = torch.tensor(self.extrapolator.derivatives(id), dtype=float32)  # Bx1

            ## exchange this log_pi_return with autoregression function ##


            ## Compute the final loss ##
            ############################
            ### loss : equation (5) ####
            ############################
            loss += - 1.0 * torch.sum(del_extrapolator * log_pi_return)              # sum(Bx1 * Bx1) -> 1
            
            ########################################################
            ### Algorithm1 step4-2 : add entropy regularier term ###
            ########################################################

            # Discourage very deterministic policies.
            if self.config.entropy_lambda > 0:
                if self.config.cont_actions:
                    entropy = torch.sum(dist_all.entropy().view(B, H, -1).sum(dim=-1) * mask) / torch.sum(mask)  # (BxH)xA -> BxH
                else:
                    log_pi_all = dist_all.view(B, H, -1)
                    pi_all = torch.exp(log_pi_all)                                      # (BxH)xA -> BxHxA
                    entropy = torch.sum(torch.sum(pi_all * log_pi_all, dim=-1) * mask) / torch.sum(mask)

                loss = loss + self.config.entropy_lambda * entropy

            # Compute the total derivative and update the parameters.

            ############################################
            ### Algorithm1 step 5 : update parameter ###
            ############################################

            self.step(loss)


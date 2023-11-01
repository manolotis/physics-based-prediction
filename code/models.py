import numpy as np

from waymo_utils.code.utils.misc import rotate_points


class CV:
    """
    # Basic cv that makes predictions for a given prediction horizon
    """

    @staticmethod
    def predict(x, y, vx, vy, t):
        """
        Given current x, y, vx, vy, and a prediction horizon t (s), return expected (x,y) at time t
        """
        x_pred = x + t * vx
        y_pred = y + t * vy

        coordinates = np.stack([x_pred, y_pred], axis=-1)

        return coordinates


class CVX:
    @staticmethod
    def predict(x, y, vx, vy, t, N=6, std=np.deg2rad(25)):
        """
        Given current x, y, vx, vy, and a prediction horizon t (s), return expected (x,y) at time t, with angular noice sampled from Normal distribution.
        Noise default from https://arxiv.org/pdf/1903.07933.pdf
        ToDo: vectorize to make more efficient
        """

        n_agents = x.shape[0]
        n_timesteps = t.shape[0]
        n_features = 2

        coordinates = np.zeros((n_agents, N, n_timesteps, n_features))

        for i_agent in range(n_agents):
            for i_prediction in range(N):
                vx_current, vy_current = vx[i_agent, 0], vy[i_agent, 0]

                # sample some noise
                noise = np.random.normal(0, std)

                # rotate current vx, vy by this noise
                velocities_original = [vx_current, vy_current]
                velocities_noisy = rotate_points(velocities_original, angle=noise)
                vx_noisy, vy_noisy = velocities_noisy

                # predict with noisy heading (except first)
                if i_prediction > 0:
                    coordinates[i_agent, i_prediction, :, 0] = x[i_agent, 0] + t * vx_noisy
                    coordinates[i_agent, i_prediction, :, 1] = y[i_agent, 0] + t * vy_noisy
                else:
                    coordinates[i_agent, i_prediction, :, 0] = x[i_agent, 0] + t * vx_current
                    coordinates[i_agent, i_prediction, :, 1] = y[i_agent, 0] + t * vy_current

        return coordinates

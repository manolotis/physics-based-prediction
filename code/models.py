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

        return x_pred, y_pred

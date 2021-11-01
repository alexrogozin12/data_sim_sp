from scipy.optimize.linesearch import scalar_search_wolfe2


class LineSearchTool(object):
    """
    Implements line search along a given direction.

    Parameters
    ----------
    method: str
        Line search strategy. Supported values:
        "Wolfe": Wolfe search.
        "Armijo": Armijo search
        "Constant": Constant step-size

    c1: coefficient for Armijo line search or first coefficient for Wolfe line search.

    c2: second coefficient for Wolfe line search.

    alpha_0: initial step-size for Wolfe and Armijo line search.

    c: step-size for Constant strategy.
    """

    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self._method == 'Constant':
            return self.c
        elif self._method == 'Armijo':
            alpha_0 = previous_alpha if previous_alpha is not None else self.alpha_0
            return self.armijo_search(oracle, x_k, d_k, alpha_0)
        elif self._method == 'Wolfe':
            alpha = scalar_search_wolfe2(
                phi=lambda step: oracle.func_directional(x_k, d_k, step),
                derphi=lambda step: oracle.grad_directional(x_k, d_k, step),
                c1=self.c1,
                c2=self.c2
            )[0]
            if alpha is None:
                return self.armijo_search(oracle, x_k, d_k, self.alpha_0)
            else:
                return alpha

        return None

    def armijo_search(self, oracle, x_k, d_k, alpha_0):
        phi = lambda step: oracle.func_directional(x_k, d_k, step)
        alpha = alpha_0
        coef = self.c1 * oracle.grad_directional(x_k, d_k, 0)
        while phi(alpha) > phi(0) + alpha * coef:
            alpha = alpha / 2
        return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()

import numpy as np

u0 = 4 * np.pi * 10**-7


class HysteresisRod:
    # TODO: is the limiting cycle B on the y-axis or is it M = B - u0*H?

    def __init__(self, br, bs, hc):
        self.br = br
        self.bs = bs
        self.hc = hc
        self.k = (1/hc) * np.tan(np.pi * br / 2 / bs)

    def b_field_top(self, h):
        return 2 * self.bs * np.arctan(self.k * (h + self.hc))/np.pi

    def b_field_bottom(self, h):
        return 2 * self.bs * np.arctan(self.k * (h - self.hc))/np.pi

    def b_field_top_derivative(self, h):
        return 2 * self.bs / (self.k*(h + self.hc)**2 + 1) / np.pi

    def b_field_bottom_derivative(self, h):
        return 2 * self.bs / (self.k*(h - self.hc)**2 + 1) / np.pi

    def mag_process_positive_h(self, h, b):
        return u0 + (self.b_field_bottom(h) - b) * (self.b_field_top_derivative(h) - u0) / \
               (self.b_field_bottom(h) - self.b_field_top(h))

    def mag_process_negative_h(self, h, b):
        return u0 + (b - self.b_field_top(h)) * (self.b_field_bottom_derivative(h) - u0) / \
               (self.b_field_bottom(h) - self.b_field_top(h))

    def plot_limiting_cycle(self, hmin, hmax):
        import matplotlib.pyplot as plt
        plt.figure()
        h = np.linspace(hmin, hmax, 1000)
        top = self.b_field_top(h)
        bottom = self.b_field_bottom(h)
        plt.plot(h, top, 'C0')
        plt.plot(h, bottom, 'C0')
        plt.axvline(x=0, color='k')
        plt.axhline(y=0, color='k')
        plt.axvline(x=self.hc, color='r', ls='dashed', alpha=0.3)
        plt.axvline(x=-self.hc, color='r', ls='dashed', alpha=0.3)
        plt.axhline(y=self.br, color='r', ls='dashed', alpha=0.3)
        plt.axhline(y=-self.br, color='r', ls='dashed', alpha=0.3)
        plt.axhline(y=self.bs, color='r', ls='dashed', alpha=0.3)
        plt.axhline(y=-self.bs, color='r', ls='dashed', alpha=0.3)
        plt.title('Limiting Hysteresis Cycle')
        plt.xlabel('H')
        plt.ylabel('B')

    def plot_limiting_cycle_derivative(self, hmin, hmax):
        import matplotlib.pyplot as plt
        plt.figure()
        h = np.linspace(hmin, hmax, 1000)
        top = self.b_field_top_derivative(h)
        bottom = self.b_field_bottom_derivative(h)
        plt.plot(h, top)
        plt.plot(h, bottom)
        plt.axvline(x=0, color='k')
        plt.title('Limiting Hysteresis Cycle Derivative')
        plt.xlabel('H')
        plt.ylabel('B')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from integrators import rk4_general

    hyst_rod = HysteresisRod(0.8, 1.5, 80)
    hyst_rod.plot_limiting_cycle(-600, 600)
    hyst_rod.plot_limiting_cycle_derivative(-600, 600)

    b = np.zeros(1000)
    h = np.linspace(0, 600, 1000)
    step = h[1] - h[0]
    for i, da in enumerate(h[:-1]):
        if h[i+1] >= h[i]:
            print('p')
            b[i + 1] = b[i] + hyst_rod.mag_process_positive_h(h[i+1], b[i]) * step
            #b[i+1] = rk4_general(hyst_rod.mag_process_positive_h, step, h[i+1], b[i])
        else:
            print('n')
            b[i + 1] = b[i] + hyst_rod.mag_process_negative_h(h[i + 1], b[i]) * step
            #b[i + 1] = rk4_general(hyst_rod.mag_process_negative_h, step, h[i + 1], b[i])

    plt.figure()
    plt.plot(h, b)
    plt.show()

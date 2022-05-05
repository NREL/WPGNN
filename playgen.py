import os
import numpy as np
from scipy.stats import truncnorm

class PLayGen():
    # Plant Layout Generator (PLayGen)
    def __init__(self, layout_style=None, N_turbs=None, D=130.,
                       spacing=None, spacing_units='D', 
                       angle=None, angle_units='degrees',
                       breaks=None, noise=None):
        super(PLayGen, self).__init__()

        # Style of wind plant to generate
        #   Must be one of ['cluster', 'single string', 'multiple string', 'parallel string']
        #   or None - in which case the style is randomly chosen
        self.layout_style = layout_style

        # The number of turbines
        self.N_turbs = N_turbs

        # Mean inter-turbine spacing and units (either 'D' for rotor diameters or 'm' for meters)
        self.spacing = spacing
        self.spacing_units = spacing_units

        # Angle for orientation of wind farm and units
        self.angle = angle
        self.angle_units = angle_units

        # List of integers describing string breaks 
        # Set to [] if no breaks are desired
        self.breaks = breaks

        # A scalar factor between [0, 1] describing string noise
        self.noise = noise

        # Rotor diameter in meters
        self.D = D

    def __call__(self):
        if self.layout_style is None:
            style = np.random.choice(['cluster', 'single string', 'multiple string', 'parallel string']) 
        else:
            style = self.layout_style
        
        if style == 'cluster':
            layout_data = self.random_cluster_layout()
        elif style == 'single string':
            layout_data = self.random_single_string_layout()
        elif style == 'multiple string':
            layout_data = self.random_multiple_string_layout()
        elif style == 'parallel string':
            layout_data = self.random_parallel_string_layout()
        else:
            assert False, "Bad wind plant layout style."

        return layout_data

    def reset(self):
        self.set_layout_style(None)
        self.set_N_turbs(None)
        self.set_spacing(None, spacing_units='D')
        self.set_angle(None, angle_units='degrees')
        self.set_breaks(None)
        self.set_noise(None)
        self.set_rotor_diameter(130.)

    def set_layout_style(self, style):
        # Set the wind plant layout style.
        if style is not None:
            valid_styles = ['cluster', 'single string', 'multiple string', 'parallel string']
            assert style.lower() in valid_styles, "Invalid layout style: {}".format(style)
            style = style.lower()
        self.layout_style = style

    def set_N_turbs(self, N_turbs):
        # Set the number of turbines. Must be integer or None. If None, N_turbs is randomly chosen for each sample.
        assert (N_turbs is None) or ((N_turbs % 1. == 0.) and (N_turbs > 0)), "Bad number of turbines."
        self.N_turbs = N_turbs

    def set_spacing(self, spacing, spacing_units=None):
        # Set the mean inter-turbine spacing and units.
        if spacing_units is not None:
            assert spacing_units in ['m', 'D'], "Bad spacing units. Must be either 'm' or 'D'."
            self.spacing_units = spacing_units

        if spacing is not None:
            assert spacing > 0., "Bad inter-turbine spacing value."
        self.spacing = spacing

    def set_angle(self, angle, angle_units=None):
        # Set the general wind plant orientation.
        if angle_units is not None:
            assert angle_units in ['degrees', 'radians'], "Bad spacing units. Must be either 'degrees' or 'radians'."
            self.angle_units = angle_units

        self.angle = angle

    def set_breaks(self, breaks):
        # Set the number of breaks in turbine strings.
        if (breaks is not None) and (not isinstance(breaks, list)):
            assert (breaks > 0.) and (breaks % 1. == 0.), "Breaks must be list of break sizes."
            breaks = [int(breaks)]
        self.breaks = breaks

    def set_noise(self, noise):
        # Set the noise level for turbine strings.
        if noise is not None:
            assert (noise > 0.) and (noise < 1.)
        self.noise = noise

    def set_rotor_diameter(self, D):
        # Set rotor diameter
        assert D > 0.
        self.D = D

    def _truncated_lognormal_(self, a, b, mu, sigma, size=1):
        mu_norm = 0.5*np.log(mu**4/(mu**2 + sigma**2))
        sigma_norm = np.sqrt(np.log(sigma**2/mu**2 + 1))
        a, b = (np.log(a) - mu_norm)/sigma_norm, (np.log(b) - mu_norm)/sigma_norm

        return np.exp(truncnorm.ppf(np.random.uniform(size=size), a, b, loc=mu_norm, scale=sigma_norm))

    def _interturbine_spacing_(self, x, y):
        # For each turbine, compute the distance to its nearest turbine (in units of self.spacing_units).

        D = (np.expand_dims(x, axis=0) - np.expand_dims(x, axis=1))**2 + \
            (np.expand_dims(y, axis=0) - np.expand_dims(y, axis=1))**2 + np.diag(np.nan*np.ones((x.size, )))
        d_turb = np.sqrt(np.nanmin(D, axis=1))

        if self.spacing_units == 'D':
            d_turb /= self.D
        
        return d_turb

    def _random_breaks_(self, N_turbs, N_breaks=None):
        # Randomly break up a turbine string with N_turbs turbines
        if N_breaks is None:
            N_breaks = np.random.randint(int(np.sqrt(N_turbs))) if N_turbs >= 5 else 0 # WAS N_turbs/5
            
        if (N_breaks > 0):
            breaks = np.random.randint(low=1, high=int(np.sqrt(N_turbs)), size=N_breaks)
        else:
            breaks = []
            
        return breaks

    def _distribute_turbines_(self, N_turbs, N_string):
        # Randomly distrbute N_turbs turbines across N_strings strings
        #   - Each string contains at least 2 turbines and no more than 75% of the total number of turbines
        max_turbs = int(0.75*N_turbs)
        N_turbs_per_string = 2*np.ones((N_string, ), dtype=np.int)
        count = 0
        while np.sum(N_turbs_per_string) < N_turbs:
            count += 1
            idx = np.random.randint(0, N_string)
            if N_turbs_per_string[idx] >= max_turbs:
                continue
            N_turbs_per_string[idx] += 1
            
        return np.flip(np.sort(N_turbs_per_string))

    def _generate_string_(self, N_turbs, R, theta, breaks, noise):
        # Generate a single of turbine locations with given properties
        
        # Insert breaks into string of turbines
        break_idx = np.ones((N_turbs, ), dtype=np.bool)
        for break_i in breaks:
            break_idx = np.insert(break_idx, np.random.randint(break_idx.size), np.zeros((break_i, ), dtype=np.bool))
        N_turbs = break_idx.size
        
        # Construct correlated noise array for to turbine locations
        noise = np.random.choice(a=noise*np.array([-.01, 0, .01]), size=(10*N_turbs, )).cumsum(0)[::10]
        noise -= noise[0]
        noise -= np.linspace(0, 1, noise.size)*noise[-1]
        
        # Build sting of turbines
        x, y = np.linspace(-1., 1., N_turbs), np.ones((N_turbs, ))

        phi = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        xy = R*(phi@np.concatenate((x.reshape((1, N_turbs)), (noise*y).reshape((1, N_turbs))), axis=0))
        x, y = xy[0, break_idx], xy[1, break_idx]
                
        return x, y

    def random_single_string_layout(self):
        # Generate a randomized single string style wind plant layout

        # Set number of turbines
        N = int(round(self._truncated_lognormal_(25, 100, 50, 20)[0])) if self.N_turbs is None else self.N_turbs
        
        # Set number and size of breaks in the string
        string_breaks = self._random_breaks_(N) if self.breaks is None else self.breaks
        
        # Set mean inter-turbine spacing in meters
        if self.spacing is None:
            d_turb = self.D*np.random.uniform(low=2.5, high=7)
        else:
            d_turb = self.spacing if self.spacing_units == 'm' else self.D*self.spacing
        
        # Set orientation of string
        if self.angle is None:
            theta = np.random.uniform(low=0., high=2.*np.pi)
        else:
            theta = self.angle if self.angle_units == 'radians' else (np.pi/180.)*self.angle

        # Set noise level for string
        noise_level = np.random.uniform(low=0.25, high=0.5) if self.noise is None else self.noise
        
        # Set the radius for a circular domain of the wind plant string
        R = 0.48*d_turb*(N+np.sum(string_breaks))
        
        # Generate string of turbines
        x, y = self._generate_string_(N, R, theta, string_breaks, noise_level)

        wind_plant = np.concatenate((x.reshape((N, 1))-np.mean(x), y.reshape((N, 1))-np.mean(y)), axis=1)

        return wind_plant

    def random_multiple_string_layout(self):
        # Generate a randomized multiple string style wind plant layout
        
        # Set number of turbines and strings
        N = int(round(self._truncated_lognormal_(25, 200, 125, 50)[0])) if self.N_turbs is None else self.N_turbs
        N_string = int(np.random.normal(loc=np.sqrt(N), scale=1))
        N_string = 2 if N_string < 2 else N_string
        
        # Set mean inter-turbine spacing in meters
        if self.spacing is None:
            d_turb = self.D*np.random.uniform(low=2.5, high=7)
        else:
            d_turb = self.spacing if self.spacing_units == 'm' else self.D*self.spacing
        
        # Set the general orientation of the wind plant
        if self.angle is None:
            theta = np.random.uniform(low=0., high=2.*np.pi)
        else:
            theta = self.angle if self.angle_units == 'radians' else (np.pi/180.)*self.angle

        # Distribute the N turbines across N_string strings
        N_per_string = self._distribute_turbines_(N, N_string)
        
        # Define the wind plant domain
        d_x, d_y = np.random.uniform(low=d_turb, high=8*self.D), np.random.uniform(low=d_turb, high=8*self.D)
        domain = 0.8*np.sqrt(N)*np.sqrt(N_string)*np.array([[-d_x, d_x], [-d_y, d_y]])

        # Set turbine locations for each string
        x, y, min_d_turb_by_str = np.zeros((0, )), np.zeros((0, )), []
        for i, N_i in enumerate(N_per_string):
            count = 0
            
            # Set number and size of breaks in the ith string
            string_breaks = self._random_breaks_(N_i) if self.breaks != [] else self.breaks
            
            # Set noise level for string
            noise_level = np.random.uniform(low=0.25, high=0.75) if self.noise is None else self.noise

            # Set the radius for a circular domain of the wind plant string
            R = 0.46*d_turb*(N_i+np.sum(string_breaks))

            # Set string orientation as perturbation to general wind plant orientation
            theta_i = truncnorm.ppf(np.random.uniform(), -1.12, 1.12, loc=theta, scale=0.5)

            # Generate string of turbines
            x_i, y_i = self._generate_string_(N_i, R, theta_i, string_breaks, noise_level)

            # Place string in the domain
            x_c = np.random.uniform(low=domain[0, 0]+R, high=domain[0, 1]-R)
            y_c = np.random.uniform(low=domain[1, 0]+R, high=domain[1, 1]-R)
            x_i_c, y_i_c = x_i + x_c, y_i + y_c

            # Make sure placement is valid. If not, replace turbine string.
            min_d_turb_by_str.append(np.min(self._interturbine_spacing_(x_i, y_i)))
            d_turb_all = self._interturbine_spacing_(np.concatenate((x, x_i_c)), np.concatenate((y, y_i_c)))
            while (x.size > 0) and any(d_turb_all < (np.min(min_d_turb_by_str)-1e-6)):
                count += 1

                x_c = np.random.uniform(low=domain[0, 0]+R, high=domain[0, 1]-R)
                y_c = np.random.uniform(low=domain[1, 0]+R, high=domain[1, 1]-R)
                x_i_c, y_i_c = x_i + x_c, y_i + y_c
                
                d_turb_all = self._interturbine_spacing_(np.concatenate((x, x_i_c)), np.concatenate((y, y_i_c)))

                # If too many failed attempts, expand domain and try again.
                if count > 100:
                    d_x += self.D
                    d_y += self.D
                    domain = 0.8*np.sqrt(N)*np.sqrt(N_string)*np.array([[-d_x, d_x], [-d_y, d_y]])

                    count = 0
        
            x, y = np.concatenate((x, x_i_c), axis=0), np.concatenate((y, y_i_c), axis=0)

        wind_plant = np.concatenate((x.reshape((N, 1))-np.mean(x), y.reshape((N, 1))-np.mean(y)), axis=1)
        
        return wind_plant

    def random_parallel_string_layout(self):
        # Generate a randomized parallel string style wind plant layout

        # Set number of turbines and strings
        N = int(round(self._truncated_lognormal_(25, 200, 125, 50)[0])) if self.N_turbs is None else self.N_turbs
        N_string = int(np.random.normal(loc=np.sqrt(N)/2., scale=1))
        N_string = 2 if N_string < 2 else N_string

        # Set mean inter-turbine spacing in meters
        if self.spacing is None:
            d_turb = self.D*np.random.uniform(low=2.5, high=7)
        else:
            d_turb = self.spacing if self.spacing_units == 'm' else self.D*self.spacing
        
        # Set the general orientation of the wind plant
        if self.angle is None:
            theta = np.random.uniform(low=0., high=2.*np.pi)
        else:
            theta = self.angle if self.angle_units == 'radians' else (np.pi/180.)*self.angle

        # Distribute the N turbines across N_string strings
        N_per_string = self._distribute_turbines_(N, N_string)
        np.random.shuffle(N_per_string)

        # Set the number and size of breaks for each turbine string
        if self.breaks == []:
            string_breaks = [[] for N_i in N_per_string]
        else:
            string_breaks = [self._random_breaks_(N_i) for N_i in N_per_string]
        
        # Identify the longest parallel string
        longest_string = int(np.max([N_i+np.sum(breaks_i) for N_i, breaks_i in zip(N_per_string, string_breaks)]))

        # Set the radius for a circular domain of the wind plant string
        R = 0.48*d_turb*longest_string
        
        # Set parallel string offset size
        offset_step = np.random.uniform(low=1., high=3.5)

        x, y = np.zeros((0, )), np.zeros((0, ))
        for i, (N_i, breaks_i) in enumerate(zip(N_per_string, string_breaks)):

            # Set noise level for string
            noise_level = np.random.uniform(low=0.25, high=0.75) if self.noise is None else self.noise

            # Set the radius for a circular domain of the wind plant string
            R_i = 0.48*d_turb*(N_i+np.sum(breaks_i))

            # Generate string of turbines
            x_i, y_i = self._generate_string_(N_i, R_i, 0., breaks_i, noise_level)
            
            # Compute horizontal/vertical offset of each parallel string
            x_i += np.random.uniform(low=R_i/2-R, high=R-R_i/2)
            if i > 0:
                y_i -= (np.max(y_i) + d_turb*(offset_step + abs(np.random.normal(loc=0, scale=0.5))))
            x, y = np.concatenate((x, x_i), axis=0), np.concatenate((y, y_i), axis=0)
            y -= np.min(y)
        
        # Center and rotate the wind plant
        phi = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        wind_plant = (phi@np.concatenate((x.reshape((1, N)) - np.mean(x), y.reshape((1, N))) - np.mean(y), axis=0)).T
        wind_plant[:, 0] -= np.mean(wind_plant[:, 0])
        wind_plant[:, 1] -= np.mean(wind_plant[:, 1])

        return wind_plant

    def random_cluster_layout(self):
        # Generate a randomized cluster style wind plant layout using a Poisson disc sampling approach

        # Set number of turbines
        N = int(round(self._truncated_lognormal_(25, 200, 125, 50)[0])) if self.N_turbs is None else self.N_turbs

        # Extra turbines used to simulate gaps in the cluster array
        N_extra = abs(int(np.random.normal(loc=np.sqrt(N), scale=np.sqrt(N))))
        
        # Set mean inter-turbine spacing in meters
        if self.spacing is None:
            d_turb = self.D*np.random.uniform(low=2.5, high=7)
        else:
            d_turb = self.spacing if self.spacing_units == 'm' else self.D*self.spacing

        d_turb_min, d_turb_max = 0.95*d_turb, 1.05*d_turb
        
        # Poisson disc algorithm for randomly generate turbines with roughly fixed spacing
        x, y = np.zeros((N+N_extra, )), np.zeros((N+N_extra, ))
        for i in range(1, N+N_extra):
            count = 0
            
            if (i%3 == 1):
                idx = np.random.randint(0, i)
            else:
                idx = i
            alpha, R = np.random.uniform(0, 2*np.pi), np.random.uniform(d_turb_min, d_turb_max)
            x_i, y_i = x[idx] + R*np.cos(alpha), y[idx] + R*np.sin(alpha)
            while np.min((x[:i] - x_i)**2 + (y[:i] - y_i)**2) < d_turb_min**2:
                alpha, R = np.random.uniform(0, 2*np.pi), np.random.uniform(d_turb_min, d_turb_max)
                x_i, y_i = x[idx] + R*np.cos(alpha), y[idx] + R*np.sin(alpha)
                
                count += 1
                if count > 10:
                    idx = np.random.randint(0, i)
            
            x[i], y[i] = x_i, y_i
        
        # Throw out the extra turbines
        if N_extra > 0:
            idx_keep = np.random.permutation(N+N_extra)[:N]
            x, y = x[idx_keep], y[idx_keep]

        # Center the wind plant at the origin
        wind_plant = np.concatenate((x.reshape((N, 1))-np.mean(x), y.reshape((N, 1))-np.mean(y)), axis=1)

        return wind_plant











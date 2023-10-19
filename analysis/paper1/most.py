"""
We adapt a scheme described by Marks and Dozier (1992, see Equations 9 - 16), in which we solve a system of equations for L, u*, H, and E iteratively, by initially assuming neutral conditions (L = ∞). We assume a constant measurement height for all measurements throughout our measurement period, even though the snow surface varys from 0 to 2 meters during the snow season. Note that our method differs from theirs in that we calculate momentum, sensible heat, and latent heat fluxes as positive AWAY from the surface.

*Marks, D., & Dozier, J. (1992). Climate and energy exchange at the snow surface in the Alpine Region of the Sierra Nevada: 2. Snow cover energy balance. Water Resources Research, 28(11), 3043–3054. https://doi.org/10.1029/92WR01483*

Also useful:
https://atmos.uw.edu/~breth/classes/AS547/lect/lect6.pdf

Section 9.1a of Brutsaert, W. (1982) Evaporation into the Atmosphere: Theory, History, and Applications. Springer, Dordrecht, 299. 
http://dx.doi.org/10.1007/978-94-017-1497-6

"""

import numpy as np

VON_KARMAN_CONSTANT = 0.40 #dimensionless
GRAVITY = 9.81 # m/s^2

# Should this just be 0 for us?
# Recommended equation
# ZERO_PLANE_DISPLACEMENT_HEIGHT = (2/3)*7.35*SNOW_SURFACE_ROUGHNESS # m
# What  I think is appropriate for snow
ZERO_PLANE_DISPLACEMENT_HEIGHT = 0

RATIO_OF_EDDY_DIFFUSIVITY_AND_VISCOSITY_FOR_HEAT = 1.0 # dimensionless
RATIO_OF_EDDY_DIFFUSIVITY_AND_VISCOSITY_FOR_WATERVAPOR = 1.0 # dimensionless
AIR_SPECIFIC_HEAT = 1005 # J / kg / K


# Is there even a point in declaring a parent class like this??? How to enforce the implementation of these functions in child classes?
class StabilityFunction:

    @staticmethod
    def mass(
        measurement_height_above_snow_surface_windspeed,
        obukhov_stability_length
    ):
        pass
    
    @staticmethod
    def heat(
        measurement_height_above_snow_surface_temperature,
        obukhov_stability_length
    ):
        pass
    
    @staticmethod
    def watervapor( 
        measurement_height_above_snow_surface_specific_humidity,
        obukhov_stability_length
    ): 
        pass

class StabilityFunctionBrutsaert1982(StabilityFunction):    
    # Helper functions
    @staticmethod
    def _stable_stability_function_generic(
        zeta # dimensionless
    ):
        if zeta > 1:
            return -5.2
        elif zeta > 0:
            return -5.2*zeta
        else:
            raise ValueError("zeta <= 0 in stable conditions??")

    @staticmethod
    def _x_func(
        zeta # dimensionless
    ):
        return (1 - 16*zeta)**(0.25)

    @staticmethod
    def _unstable_stability_function_heat_and_watervapor(
        zeta # dimensionless
    ):
        return 2*np.log(
            (1 + StabilityFunctionBrutsaert1982._x_func(zeta)**2) / 2)

    @staticmethod
    def mass(
        measurement_height_above_snow_surface_windspeed,
        obukhov_stability_length
    ):
        zeta = measurement_height_above_snow_surface_windspeed / obukhov_stability_length
        if zeta > 0:
            return  StabilityFunctionBrutsaert1982._stable_stability_function_generic(zeta)
        else:
            return 2*np.log(
                (1 + StabilityFunctionBrutsaert1982._x_func(zeta)) / 2
            ) + np.log(
                (1 + StabilityFunctionBrutsaert1982._x_func(zeta)**2) / 2
            ) - 2*np.arctan(
                StabilityFunctionBrutsaert1982._x_func(zeta)
            ) + np.pi/2

    @staticmethod
    def heat(
        measurement_height_above_snow_surface_temperature,
        obukhov_stability_length
    ):
        zeta = measurement_height_above_snow_surface_temperature / obukhov_stability_length
        if zeta > 0:
            return StabilityFunctionBrutsaert1982._stable_stability_function_generic(zeta)
        else:
            return StabilityFunctionBrutsaert1982._unstable_stability_function_heat_and_watervapor(zeta)

    @staticmethod
    def watervapor(
        measurement_height_above_snow_surface_specific_humidity,
        obukhov_stability_length
    ):
        zeta = measurement_height_above_snow_surface_specific_humidity / obukhov_stability_length
        if zeta > 0:
            return StabilityFunctionBrutsaert1982._stable_stability_function_generic(zeta)
        else:
            return StabilityFunctionBrutsaert1982._unstable_stability_function_heat_and_watervapor(zeta)

class  StabilityFunctionHoltslagDeBruin(StabilityFunction):    
    
    @staticmethod
    def _generic(
            measurement_height_above_snow_surface,
            obukhov_stability_length
    ):
        zeta = measurement_height_above_snow_surface / obukhov_stability_length
        a = 0.7
        b = 0.75
        c = 5
        d = 0.35
        return -a*zeta - b*(zeta - c/d)*np.exp(-d*zeta) - b*c/d
    
    @staticmethod        
    def mass(
        measurement_height_above_snow_surface_windspeed,
        obukhov_stability_length
    ):
        return StabilityFunctionHoltslagDeBruin._generic(
            measurement_height_above_snow_surface_windspeed,
            obukhov_stability_length
        )
    
    @staticmethod    
    def heat(
        measurement_height_above_snow_surface_temperature,
        obukhov_stability_length
    ):
        return StabilityFunctionHoltslagDeBruin._generic(
            measurement_height_above_snow_surface_temperature,
            obukhov_stability_length
        )
    
    @staticmethod    
    def watervapor(
        measurement_height_above_snow_surface_specific_humidity,
        obukhov_stability_length
    ): 
        return StabilityFunctionHoltslagDeBruin._generic(
            measurement_height_above_snow_surface_specific_humidity,
            obukhov_stability_length
        )
    
class StabilityFunctionStearnsWeidner(StabilityFunction):    

    @staticmethod
    def _generic(
            measurement_height_above_snow_surface,
            obukhov_stability_length
    ):
        zeta = measurement_height_above_snow_surface / obukhov_stability_length
        Y = (1 + 5*zeta)**0.5
        return np.log(1 + Y)**2 - 2*Y - 2*Y**3/3 + 1.2804
        
    @staticmethod
    def mass(
        measurement_height_above_snow_surface_windspeed,
        obukhov_stability_length
    ):
        return StabilityFunctionStearnsWeidner._generic(
            measurement_height_above_snow_surface_windspeed,
            obukhov_stability_length
        )
    
    @staticmethod
    def heat(
        measurement_height_above_snow_surface_temperature,
        obukhov_stability_length
    ):
        return StabilityFunctionStearnsWeidner._generic(
            measurement_height_above_snow_surface_temperature,
            obukhov_stability_length
        )
    
    @staticmethod
    def watervapor( 
        measurement_height_above_snow_surface_specific_humidity,
        obukhov_stability_length
    ): 
        return StabilityFunctionStearnsWeidner._generic(
            measurement_height_above_snow_surface_specific_humidity,
            obukhov_stability_length
        )

    
#     def stability_function_stearns_and_weidner(
#             self,
#             measurement_height_above_snow_surface_specific_humidity,
#             obukhov_stability_length
#     ):

## Class to hold the 7 functions that require solving
class MOST:

    _stability_function = None
    _stability_function_heat = None
    _stability_function_watervapor = None
    _snow_surface_roughness = None

    MAX_ITERATIONS = 50

    def __init__(
        self,
        stab_class: StabilityFunction,
        snow_surface_roughness: float = 1e-4 # m, Marks and Dozier report its in the range [1e-4, 5e-3]
    ):
        self._stability_function = stab_class.mass
        self._stability_function_heat = stab_class.heat
        self._stability_function_watervapor = stab_class.watervapor
        self._snow_surface_roughness = snow_surface_roughness


    def solve(
            self,
            wind_speed_ls, # m/s
            potential_temperature_ls, # kelvin
            surface_potential_temperature_ls, # kelvin
            air_density_ls, # ???
            specific_humidity_ls, # g/kg
            surface_specific_humidity_ls, # g/kg
            temperature_ls, # kelvin
            measurement_height_ls # m
    ):
        assert (
            self._stability_function is not None and 
            self._stability_function_heat is not None and 
            self._stability_function_watervapor is not None
        )
        assert (
            len(wind_speed_ls) == 
            len(potential_temperature_ls) == 
            len(surface_potential_temperature_ls) == 
            len(air_density_ls) == 
            len(specific_humidity_ls) == 
            len(surface_specific_humidity_ls) == 
            len(temperature_ls) == 
            len(measurement_height_ls)
        )
        # Instantiate lists to hold solutions for each time step
        L_solutions = []
        u_friction_solutions = [] 
        H_solutions = [] 
        E_solutions = [] 

        # Instantiate list to hold, for each timesteps, how many iterations it took to converge
        iteration_finished_list = []

        for idx in range(0, len(wind_speed_ls)):
            wind_speed = wind_speed_ls[idx]
            potential_temperature = potential_temperature_ls[idx]
            surface_potential_temperature = surface_potential_temperature_ls[idx]
            air_density = air_density_ls[idx]
            specific_humidity = specific_humidity_ls[idx]
            surface_specific_humidity = surface_specific_humidity_ls[idx]
            temperature = temperature_ls[idx]
            measurement_height = measurement_height_ls[idx]

            # if any nans, assign nans for all model results
            if (
                np.isnan(wind_speed) or
                np.isnan(potential_temperature) or
                np.isnan(surface_potential_temperature) or
                np.isnan(air_density) or
                np.isnan(specific_humidity) or
                np.isnan(surface_specific_humidity) or
                np.isnan(temperature) or
                np.isnan(measurement_height)
            ):
                L_solutions.append(np.nan)
                u_friction_solutions.append(np.nan)
                H_solutions.append(np.nan)
                E_solutions.append(np.nan)
            else:
                # Instantiate lists to hold iterative solutions for this single set of inputs
                L_current_iterations = []
                u_friction_current_iterations = [] 
                H_current_iterations = [] 
                E_current_iterations = [] 
        
                # Assign the initial guess for L (neutral conditions)
                L =  float("inf")
                
                # Perform the iterative solution, break when two consecutive results are np.close
                for i in range(0, self.MAX_ITERATIONS):
                    # Print a message if we have reached the maximum iterations (our solution likely has not converged)
                    if i == self.MAX_ITERATIONS-1:
                        print("Reached maximum iterations")

                    # Calculate all the solutions
                    Phi_m = self._stability_function(measurement_height, L)
                    Phi_H = self._stability_function_heat(measurement_height, L)
                    Phi_E = self._stability_function_watervapor(measurement_height, L)
                    u_friction = self.friction_velocity(wind_speed, measurement_height, Phi_m)
                    H = self.sensible_heat_flux(
                        potential_temperature, 
                        surface_potential_temperature, 
                        u_friction, 
                        air_density, 
                        measurement_height, 
                        Phi_H
                    )
                    E = self.latent_heat_flux(
                        specific_humidity, 
                        surface_specific_humidity, 
                        u_friction, 
                        air_density, 
                        measurement_height, 
                        Phi_E
                    )
                    # (and update L)
                    L = self.obukhov_stability_length(
                        u_friction, 
                        air_density, 
                        H, 
                        temperature, 
                        E
                    )

                    # Append our solutions to the iterative solutions lists
                    L_current_iterations.append(L)
                    u_friction_current_iterations.append(u_friction)
                    H_current_iterations.append(H)
                    E_current_iterations.append(E)

                    # If our iterative solutions have converged (the last two solutions are close enough, using the 1e-8 absolute and relative tolerances), stop iterating
                    if len(L_current_iterations) > 1 and np.isclose(L_current_iterations[-2], L_current_iterations[-1], rtol=1e-08, atol=1e-08):
                        iteration_finished_list.append(i)
                        break
                
                L_solutions.append(L_current_iterations[-1])
                u_friction_solutions.append(u_friction_current_iterations[-1])
                H_solutions.append(H_current_iterations[-1])
                E_solutions.append(E_current_iterations[-1])

        return L_solutions, u_friction_solutions, H_solutions, E_solutions
        
    
    def obukhov_stability_length(
        self,
        friction_velocity, # m/s 
        air_density, # kg/m^3 
        sensible_heat_flux, # W/m^2
        air_temperature, # K
        latent_heat_flux # kg/m^2/s
    ):
        """
        Returns Obukhov stability length, units of m.
        """
        return - (
            (friction_velocity**3)*air_density
        ) / (
            VON_KARMAN_CONSTANT * GRAVITY * (
                (sensible_heat_flux / (air_temperature * AIR_SPECIFIC_HEAT))
                +
                0.61 * latent_heat_flux
            )
        )

    def friction_velocity(
        self,
        wind_speed, # m/s 
        measurement_height_above_snow_surface_windspeed, # of wind speed measurement, m
        stability
    ):
        """
        Returns friction velocity, units of m/s.
        """
        return (
            wind_speed * VON_KARMAN_CONSTANT
        ) / (
            np.log(
                (measurement_height_above_snow_surface_windspeed - ZERO_PLANE_DISPLACEMENT_HEIGHT) / self._snow_surface_roughness
            )
            -
            stability
        )

    def sensible_heat_flux(
        self,
        air_potential_temperature, # K
        snow_surface_potential_temperature, # K
        friction_velocity, # m/s
        air_density, # kg/m^3
        measurement_height_above_snow_surface_temperature, # of temperature measurement, m
        stability # m
    ):
        """
        Returns sensible heat flux, positive away from the surface, units of W/m^2.
        """
        return (
            (snow_surface_potential_temperature - air_potential_temperature)
            *
            RATIO_OF_EDDY_DIFFUSIVITY_AND_VISCOSITY_FOR_HEAT
            *
            VON_KARMAN_CONSTANT*friction_velocity*air_density*AIR_SPECIFIC_HEAT
        ) / (
            np.log(
                (measurement_height_above_snow_surface_temperature - ZERO_PLANE_DISPLACEMENT_HEIGHT) / self._snow_surface_roughness
            )
            -
            stability
        )

    def latent_heat_flux(
        self,
        air_specific_humidity, # g / kg
        snow_surface_specific_humidity, # g / kg
        friction_velocity, # m/s
        air_density, # kg/m^3
        measurement_height_above_snow_surface_specific_humidity, # of specific humidity measurement
        stability
    ):
        """
        Returns latent heat flux, positive away from the surface, units of kg/m^2/s.
        """
        return (
            (snow_surface_specific_humidity - air_specific_humidity)
            *
            RATIO_OF_EDDY_DIFFUSIVITY_AND_VISCOSITY_FOR_WATERVAPOR
            *
            VON_KARMAN_CONSTANT*friction_velocity*air_density
        ) / (
            np.log(
                (measurement_height_above_snow_surface_specific_humidity - ZERO_PLANE_DISPLACEMENT_HEIGHT) / self._snow_surface_roughness
            )
            -
            stability
        )
    
class MOSTMulti():

    _stability_function = None
    _stability_function_heat = None
    _stability_function_watervapor = None

    MAX_ITERATIONS = 50

    def __init__(
        self,
        stab_class: StabilityFunction
    ):
        self._stability_function = stab_class.mass
        self._stability_function_heat = stab_class.heat
        self._stability_function_watervapor = stab_class.watervapor    

    def solve(
                self,
                wind_speed_1_ls, # m/s
                wind_speed_2_ls, # m/s
                potential_temperature_1_ls, # kelvin
                potential_temperature_2_ls, # kelvin
                air_density_ls, # ???
                specific_humidity_1_ls, # g/kg
                specific_humidity_2_ls, # g/kg
                temperature_ls, # kelvin
                measurement_height_1_ls, # m
                measurement_height_2_ls # m
        ):
            assert (
                self._stability_function is not None and 
                self._stability_function_heat is not None and 
                self._stability_function_watervapor is not None
            )
            assert (
                len(wind_speed_1_ls) == 
                len(wind_speed_2_ls) == 
                len(potential_temperature_1_ls) == 
                len(potential_temperature_2_ls) == 
                len(air_density_ls) == 
                len(specific_humidity_1_ls) == 
                len(specific_humidity_2_ls) == 
                len(temperature_ls) == 
                len(measurement_height_1_ls) == len(measurement_height_2_ls)
            )
            # Instantiate lists to hold solutions for each time step
            L_solutions = []
            u_friction_solutions = [] 
            H_solutions = [] 
            E_solutions = [] 

            # Instantiate list to hold, for each timesteps, how many iterations it took to converge
            iteration_finished_list = []

            for idx in range(0, len(wind_speed_1_ls)):
                wind_speed_1 = wind_speed_1_ls[idx]
                wind_speed_2 = wind_speed_2_ls[idx]
                potential_temperature_1 = potential_temperature_1_ls[idx]
                potential_temperature_2 = potential_temperature_2_ls[idx]
                air_density = air_density_ls[idx]
                specific_humidity_1 = specific_humidity_1_ls[idx]
                specific_humidity_2 = specific_humidity_2_ls[idx]
                temperature = temperature_ls[idx]
                measurement_height_1 = measurement_height_1_ls[idx]
                measurement_height_2 = measurement_height_2_ls[idx]

                # if any nans, assign nans for all model results
                if (
                    np.isnan(wind_speed_1) or
                    np.isnan(wind_speed_2) or
                    np.isnan(potential_temperature_1) or
                    np.isnan(potential_temperature_2) or
                    np.isnan(air_density) or
                    np.isnan(specific_humidity_1) or
                    np.isnan(specific_humidity_2) or
                    np.isnan(temperature) or
                    np.isnan(measurement_height_1) or
                    np.isnan(measurement_height_2)
                ):
                    L_solutions.append(np.nan)
                    u_friction_solutions.append(np.nan)
                    H_solutions.append(np.nan)
                    E_solutions.append(np.nan)
                else:
                    # Instantiate lists to hold iterative solutions for this single set of inputs
                    L_current_iterations = []
                    u_friction_current_iterations = [] 
                    H_current_iterations = [] 
                    E_current_iterations = [] 
            
                    # Assign the initial guess for L (neutral conditions)
                    L =  float("inf")
                    
                    # Perform the iterative solution, break when two consecutive results are np.close
                    for i in range(0, self.MAX_ITERATIONS):
                        # Print a message if we have reached the maximum iterations (our solution likely has not converged)
                        if i == self.MAX_ITERATIONS-1:
                            print("Reached maximum iterations")

                        # Calculate all the solutions
                        Phi_m_1 = self._stability_function(             measurement_height_1, L)
                        Phi_m_2 = self._stability_function(             measurement_height_2, L)
                        Phi_H_1 = self._stability_function_heat(        measurement_height_1, L)
                        Phi_H_2 = self._stability_function_heat(        measurement_height_2, L)
                        Phi_E_1 = self._stability_function_watervapor(  measurement_height_1, L)
                        Phi_E_2 = self._stability_function_watervapor(  measurement_height_2, L)


                        u_friction = self.friction_velocity(
                            wind_speed_1, 
                            wind_speed_2, 
                            measurement_height_1, 
                            measurement_height_2, 
                            L,
                            Phi_m_1,
                            Phi_m_2
                        )
                        H = self.sensible_heat_flux(
                            potential_temperature_1,
                            potential_temperature_2,
                            u_friction,
                            air_density,
                            measurement_height_1, 
                            measurement_height_2,
                            L,
                            Phi_H_1,
                            Phi_H_2
                        )
                        E = self.latent_heat_flux(
                            specific_humidity_1,
                            specific_humidity_2,
                            u_friction,
                            air_density,
                            measurement_height_1,
                            measurement_height_2,
                            L,
                            Phi_E_1,
                            Phi_E_2
                        )
                        # (and update L)
                        L = self.obukhov_stability_length(
                            u_friction, 
                            air_density, 
                            H, 
                            temperature, 
                            E
                        )

                        # Append our solutions to the iterative solutions lists
                        L_current_iterations.append(L)
                        u_friction_current_iterations.append(u_friction)
                        H_current_iterations.append(H)
                        E_current_iterations.append(E)

                        # If our iterative solutions have converged (the last two solutions are close enough, using the 1e-8 absolute and relative tolerances), stop iterating
                        if len(L_current_iterations) > 1 and np.isclose(L_current_iterations[-2], L_current_iterations[-1], rtol=1e-08, atol=1e-08):
                            iteration_finished_list.append(i)
                            break
                    
                    L_solutions.append(L_current_iterations[-1])
                    u_friction_solutions.append(u_friction_current_iterations[-1])
                    H_solutions.append(H_current_iterations[-1])
                    E_solutions.append(E_current_iterations[-1])

            return L_solutions, u_friction_solutions, H_solutions, E_solutions

    def obukhov_stability_length(
        self,
        friction_velocity, # m/s 
        air_density, # kg/m^3 
        sensible_heat_flux, # W/m^2
        air_temperature, # K
        latent_heat_flux # kg/m^2/s
    ):
        """
        Returns Obukhov stability length, units of m.
        """
        return - (
            (friction_velocity**3)*air_density
        ) / (
            VON_KARMAN_CONSTANT * GRAVITY * (
                (sensible_heat_flux / (air_temperature * AIR_SPECIFIC_HEAT))
                +
                0.61 * latent_heat_flux
            )
        )

    def friction_velocity(
        self,
        wind_speed_1, # m/s 
        wind_speed_2, # m/s 
        measurement_height_above_snow_surface_windspeed_1, # of wind speed measurement, m
        measurement_height_above_snow_surface_windspeed_2, # of wind speed measurement, m
        obukhov_length,
        stability_1,
        stability_2
    ):
        """
        Returns friction velocity, units of m/s.
        """
        return (
            (wind_speed_2 - wind_speed_1) * VON_KARMAN_CONSTANT
        ) / (
            np.log(
                (measurement_height_above_snow_surface_windspeed_2) 
                / (measurement_height_above_snow_surface_windspeed_1)
            ) - stability_2 + stability_1
        )

    def sensible_heat_flux(
        self,
        potential_temperature_1, # K
        potential_temperature_2, # K
        friction_velocity, # m/s
        air_density, # kg/m^3
        measurement_height_above_snow_surface_temperature_1, # of temperature measurement, m
        measurement_height_above_snow_surface_temperature_2, # of temperature measurement, m
        obukhov_length,
        stability_1,
        stability_2
    ):
        """
        Returns sensible heat flux, positive away from the surface, units of W/m^2.
        """
        return (
            (potential_temperature_1 - potential_temperature_2)
            *
            RATIO_OF_EDDY_DIFFUSIVITY_AND_VISCOSITY_FOR_HEAT
            *
            VON_KARMAN_CONSTANT*friction_velocity*air_density*AIR_SPECIFIC_HEAT
        ) / (
            np.log(
                (measurement_height_above_snow_surface_temperature_2) 
                / (measurement_height_above_snow_surface_temperature_1)
            ) - stability_2 + stability_1
        )

    def latent_heat_flux(
        self,
        specific_humidity_1, # g / kg
        specific_humidity_2, # g / kg
        friction_velocity, # m/s
        air_density, # kg/m^3
        measurement_height_above_snow_surface_specific_humidity_1, # of specific humidity measurement
        measurement_height_above_snow_surface_specific_humidity_2, # of specific humidity measurement
        obukhov_length,
        stability_1,
        stability_2
    ):
        """
        Returns latent heat flux, positive away from the surface, units of kg/m^2/s.
        """
        return (
            (specific_humidity_1 - specific_humidity_2)
            *
            RATIO_OF_EDDY_DIFFUSIVITY_AND_VISCOSITY_FOR_WATERVAPOR
            *
            VON_KARMAN_CONSTANT*friction_velocity*air_density
        ) / (
            np.log(
                (measurement_height_above_snow_surface_specific_humidity_2) 
                / (measurement_height_above_snow_surface_specific_humidity_1)
            ) - stability_2 + stability_1
        )
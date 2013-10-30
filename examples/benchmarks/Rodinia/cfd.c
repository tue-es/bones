//
// This file is part of the Bones source-to-source compiler examples. This C-code
// demonstrates the use of Bones for an example application: 'Unstructured Grid-
// Based CFD Solvers', taken from the Rodinia benchmark suite. For more information
// on the application or on Bones please use the contact information below.
//
// == More information on unstructured grid based CFD solvers:
// Website............http://web.cos.gmu.edu/~acorriga/pubs/gpu_cfd/
// Article............http://web.cos.gmu.edu/~acorriga/pubs/gpu_cfd/aiaa_2009_4001.pdf
// Original code......https://www.cs.virginia.edu/~skadron/wiki/rodinia/
//
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........applications/cfd.c
// Authors............Cedric Nugteren
// Original author....Andrew Corrigan
// Last modified on...10-Aug-2012
//

//########################################################################
//### Includes
//########################################################################

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//########################################################################
//### Data types
//########################################################################

typedef struct {
	float x;
	float y;
	float z;
} float3;

//########################################################################
//### Forward declarations
//########################################################################

inline void compute_flux_contribution(float3 momentum, float density_energy, float pressure, float3 velocity, float3 *fc_momentum_x, float3 *fc_momentum_y, float3 *fc_momentum_z, float3 *fc_density_energy);

//########################################################################
//### Options
//########################################################################

#define GAMMA 1.4f
#define iterations 2000
#define NNB 4
#define RK 3	// 3rd order RK

#define FF_MACH 1.2f
#define DEG_ANGLE_OF_ATTACK 0.0f
#define NDIM 3

//########################################################################
//### Defines
//########################################################################

#define VAR_DENSITY 0
#define VAR_MOMENTUM 1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)

//########################################################################
//### Global variables
//########################################################################

float ff_variable[NVAR];
float3 ff_flux_contribution_momentum_x;
float3 ff_flux_contribution_momentum_y;
float3 ff_flux_contribution_momentum_z;
float3 ff_flux_contribution_density_energy;

//########################################################################
//### Start of the main function
//########################################################################

int main(void) {
	
	// Declare the loop iterators
	int i,j;
	
	// Declare far field variables
	const float angle_of_attack = (M_PI/180.0f) * DEG_ANGLE_OF_ATTACK;
	float ff_pressure, ff_speed_of_sound, ff_speed;
	float3 ff_velocity, ff_momentum;
	
	// Declare other domain variables
	
	// Declare other/helper variables
	
	// Compute the far field
	printf("\n[cfd] Set the far field conditions"); fflush(stdout);
	{
		ff_variable[VAR_DENSITY] = 1.4f;
		ff_pressure = 1.0f;
		ff_speed_of_sound = sqrt(GAMMA*ff_pressure / ff_variable[VAR_DENSITY]);
		ff_speed = FF_MACH*ff_speed_of_sound;
		
		// Compute the velocity
		ff_velocity.x = ff_speed*cos(angle_of_attack);
		ff_velocity.y = ff_speed*sin(angle_of_attack);
		ff_velocity.z = 0.0f;
		
		// Update the variable
		ff_variable[VAR_MOMENTUM+0] = ff_variable[VAR_DENSITY] * ff_velocity.x;
		ff_variable[VAR_MOMENTUM+1] = ff_variable[VAR_DENSITY] * ff_velocity.y;
		ff_variable[VAR_MOMENTUM+2] = ff_variable[VAR_DENSITY] * ff_velocity.z;
		ff_variable[VAR_DENSITY_ENERGY] = ff_variable[VAR_DENSITY]*0.5f*ff_speed*ff_speed + (ff_pressure/(GAMMA-1.0f));
		
		// Set the momentum
		ff_momentum.x = ff_variable[VAR_MOMENTUM+0];
		ff_momentum.y = ff_variable[VAR_MOMENTUM+1];
		ff_momentum.z = ff_variable[VAR_MOMENTUM+2];
		
		// Compute the flux contribution
		compute_flux_contribution(
			ff_momentum,
			ff_variable[VAR_DENSITY_ENERGY],
			ff_pressure,
			ff_velocity,
			&ff_flux_contribution_momentum_x,
			&ff_flux_contribution_momentum_y,
			&ff_flux_contribution_momentum_z,
			&ff_flux_contribution_density_energy
		);
	}
	
	// Initialising memory
	printf("\n[cfd] Initialising memory"); fflush(stdout);
	
	
	// Clean-up and exit
	printf("\n[cfd] Completed\n\n"); fflush(stdout);
	fflush(stdout);
	return 0;
}

//########################################################################
//### Function to compute the flux contribution
//########################################################################

inline void compute_flux_contribution(
		float3 momentum,
		float density_energy,
		float pressure,
		float3 velocity,
		float3 *fc_momentum_x,
		float3 *fc_momentum_y,
		float3 *fc_momentum_z,
		float3 *fc_density_energy
	) {
	
	// Compute the x-momentum
	(*fc_momentum_x).x = velocity.x*momentum.x + pressure;
	(*fc_momentum_x).y = velocity.x*momentum.y;
	(*fc_momentum_x).z = velocity.x*momentum.z;
	
	// Compute the y-momentum
	(*fc_momentum_y).x = velocity.x*momentum.y;
	(*fc_momentum_y).y = velocity.y*momentum.y + pressure;
	(*fc_momentum_y).z = velocity.y*momentum.z;
	
	// Compute the z-momentum
	(*fc_momentum_z).x = velocity.x*momentum.z;
	(*fc_momentum_z).y = velocity.y*momentum.z;
	(*fc_momentum_z).z = velocity.z*momentum.z + pressure;

	// Compute energy density
	(*fc_density_energy).x = velocity.x*density_energy+pressure;
	(*fc_density_energy).y = velocity.y*density_energy+pressure;
	(*fc_density_energy).z = velocity.z*density_energy+pressure;
}

//########################################################################
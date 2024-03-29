#pragma once

#include <vector>
#include <thread>

#include "GL/glew.h"
#include <glm/glm.hpp>

#include "Particle.h"
#include "Geometry.h"

#define THREAD_COUNT 8

class SPHSystem
{
private:
	//particle data
	unsigned int numParticles;

	bool started;

	//initializes the particles that will be used
	void initParticles();

	// Creates hash table for particles in infinite domain
	void buildTable();

	// Sphere geometry for rendering
	Geometry* sphere;
	glm::mat4 sphereScale;
	glm::mat4* sphereModelMtxs;
	GLuint vbo;

	// Threads and thread blocks
	std::thread threads[THREAD_COUNT];
	int blockBoundaries[THREAD_COUNT + 1];
	int tableBlockBoundaries[THREAD_COUNT + 1];

public:
	SPHSystem(unsigned int numParticles, float mass, float restDensity, float gasConst, float viscosity, float h, float g, float tension);
	~SPHSystem();
	std::vector<std::vector<Particle*>> neighbouringParticles;
	//kernel/fluid constants
	float POLY6, SPIKY_GRAD, SPIKY_LAP, GAS_CONSTANT, MASS, H2, SELF_DENS;

	//fluid properties
	float restDensity;
	float viscosity, h, g, tension;
	float bound;
	std::vector<Particle*> particles;
	Particle** particleTable;
	glm::ivec3 getCell(Particle *p) const;
	
	//void surfacetension();

	// std::mutex mtx;
	void updateNeighbor();
	//helper
	double W_poly6(glm::vec3 i, glm::vec3 j, double h);
	double W_spikygrad(glm::vec3 i, glm::vec3 j, double h);
	//double W_spline(double r, double h);
	void positionCorrection(float deltatime);
	//updates the SPH system
	void update(float deltaTime);

	//draws the SPH system & its particles
	void draw(const glm::mat4& viewProjMtx, GLuint shader);

	void reset();
	void startSimulation();
};


#pragma once
#include "GL/glew.h"
#include <glm/glm.hpp>

class Particle
{
private:
	//for creating ids
	static int pCount;
public:
	// Attributes of particle
	float mass, size, elasticity;
	double lamda;
	glm::vec3 position, velocity, acceleration;
	glm::vec3 force;
	glm::vec3 normal_n;
	glm::vec3 normal;
	glm::vec3 prev;

	// For linked list
	Particle* next;
	int N;
	float density;
	float pressure;
	float id;
	float color;
	Particle(float mass, float size, glm::vec3 position, glm::vec3 velocity);
	~Particle();
};


#include "SPHSystem.h"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/norm.hpp>

#define PI 3.14159265f
#define TABLE_SIZE 1000000
#define EPS 0.0001f
// This will lock across all instances of SPHSystem's,
// however since we only really have one instance, this
// should be okay for now  
std::mutex mtx;
int MAX_ITERATION = 20;
std::vector<glm::vec3>pre_pos; 
/**
 * Hashes the position of a cell, giving where
 * the cell goes in the particle table 
 */
uint getHash(const glm::ivec3& cell) {
	return (
		(uint)(cell.x * 73856093) 
	  ^ (uint)(cell.y * 19349663) 
	  ^ (uint)(cell.z * 83492791)
	) % TABLE_SIZE;
}

SPHSystem::SPHSystem(unsigned int numParticles, float mass, float restDensity, float gasConst, float viscosity, float h, float g, float tension) {
	this->numParticles = numParticles;
	this->restDensity = restDensity;
	this->viscosity = viscosity;
	this->h = h;//constant to measure how close particles can be
	this->g = g;
	this->tension = tension;
	this->bound = 2.0f;
	POLY6 = 315.0f / (64.0f * PI * pow(h, 9));
	SPIKY_GRAD = -45.0f / (PI * pow(h, 6));
	SPIKY_LAP = 45.0f / (PI * pow(h, 6));
	MASS = mass;
	GAS_CONSTANT = gasConst;
	H2 = h * h;
	SELF_DENS = MASS * POLY6 * pow(h, 6);

	//setup densities & volume
	int cbNumParticles = numParticles * numParticles * numParticles;
	neighbouringParticles.resize(cbNumParticles);
	particles.resize(cbNumParticles);

	//initialize particles
	initParticles();

	// Load in sphere geometry and allocate matrice space
	sphere = new Geometry("resources/lowsphere.obj");
	sphereScale = glm::scale(glm::vec3(h/2.f));
	sphereModelMtxs = new glm::mat4[cbNumParticles];
	
	// Generate VBO for sphere model matrices
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(glm::mat4), &sphereModelMtxs[0], GL_DYNAMIC_DRAW);

	// Setup instance VAO
	glBindVertexArray(sphere->vao);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), 0);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)sizeof(glm::vec4));
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(sizeof(glm::vec4)*2));
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(sizeof(glm::vec4)*3));

	glVertexAttribDivisor(2,1);
	glVertexAttribDivisor(3,1);
	glVertexAttribDivisor(4,1);
	glVertexAttribDivisor(5,1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//start init
	started = false;

	// Allocate table memory
	particleTable = (Particle **)malloc(sizeof(Particle *) * TABLE_SIZE);

	// Init block boundaries (for funcs that loop through particles)
	blockBoundaries[0] = 0;
	int blockSize = particles.size() / THREAD_COUNT;
	for (int i = 1; i < THREAD_COUNT; i++) {
		blockBoundaries[i] = i * blockSize;
	}
	blockBoundaries[THREAD_COUNT] = particles.size();

	// Init table block boundaries (for table clearing func)
	tableBlockBoundaries[0] = 0;
	blockSize = TABLE_SIZE / THREAD_COUNT;
	for (int i = 1; i < THREAD_COUNT; i++) {
		tableBlockBoundaries[i] = i * blockSize;
	}
	tableBlockBoundaries[THREAD_COUNT] = TABLE_SIZE;
}

SPHSystem::~SPHSystem() {
	// free table
	free(particleTable);
	free(sphereModelMtxs);

	//delete particles
	particles.clear();
	particles.shrink_to_fit();

	//delete neighbouring particles
	neighbouringParticles.clear();
	neighbouringParticles.shrink_to_fit();
}

void SPHSystem::initParticles() {
	std::srand(1024);
	float particleSeperation = h + 0.01f;
	for (int i = 0; i < numParticles; i++) {
		for (int j = 0; j < numParticles; j++) {
			for (int k = 0; k < numParticles; k++) {
				// dam like particle positions
				float ranX = (float(rand()) / float((RAND_MAX)) * 0.5f - 1) * h / 10;
				float ranY = (float(rand()) / float((RAND_MAX)) * 0.5f - 1) * h / 10;
				float ranZ = (float(rand()) / float((RAND_MAX)) * 0.5f - 1) * h / 10;
				glm::vec3 nParticlePos = glm::vec3(i * particleSeperation + ranX - 1.5f, j * particleSeperation + ranY + h + 0.1f, k * particleSeperation + ranZ - 1.5f);

				//create new particle
				Particle* nParticle = new Particle(MASS, h,	nParticlePos, glm::vec3(0));

				//append particle
				particles[i + (j + numParticles * k) * numParticles] = nParticle;
			}
		}
	}
}

/**
 * Parallel computation function for calculating density
 * and pressures of particles in the given SPH System.
 */
void parallelDensityAndPressures(const SPHSystem& sphSystem, int start, int end) {
	float massPoly6Product = sphSystem.MASS * sphSystem.POLY6;
	
	for (int i = start; i < end; i++) {
		float pDensity = 0;
		Particle* pi = sphSystem.particles[i];
		glm::ivec3 cell = sphSystem.getCell(pi);

		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
					uint index = getHash(near_cell);
					Particle* pj = sphSystem.particleTable[index];
					
					// Iterate through cell linked list
					while (pj != NULL) {
						float dist2 = glm::length2(pj->position - pi->position);
						if (dist2 < sphSystem.H2 && pi != pj) {
							pDensity += massPoly6Product * glm::pow(sphSystem.H2 - dist2, 3);
						}
						pj = pj->next;
					}
				}
			}
		}
		
		// Include self density (as itself isn't included in neighbour)
		pi->density = pDensity + sphSystem.SELF_DENS;

		// Calculate pressure
		float pPressure = sphSystem.GAS_CONSTANT * (pi->density - sphSystem.restDensity);
		pi->pressure = pPressure;


	}
	// update color field
	//float massSpiky=sphSystem.MASS*sphSystem.s
	//for (int i = start; i < end; i++) {
	//	float pColor = 0;
	//	Particle* pi = sphSystem.particles[i];
	//	glm::ivec3 cell = sphSystem.getCell(pi);

	//	for (int x = -1; x <= 1; x++) {
	//		for (int y = -1; y <= 1; y++) {
	//			for (int z = -1; z <= 1; z++) {
	//				glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
	//				uint index = getHash(near_cell);
	//				Particle* pj = sphSystem.particleTable[index];

	//				// Iterate through cell linked list
	//				while (pj != NULL) {
	//					float dist2 = glm::length2(pj->position - pi->position);
	//					if (dist2 < sphSystem.H2 && pi != pj) {
	//						pColor += massPoly6Product * glm::pow(sphSystem.H2 - dist2, 3)/pj->density;
	//						//pColor /= pj->density;
	//					}
	//					pj = pj->next;
	//				}
	//			}
	//		}
	//	}
	//	pi->color = pColor+ massPoly6Product * glm::pow(sphSystem.H2, 3) / pi->density;
	//}
	//update normal
	for (int i = start; i < end; i++) {
		glm::vec3 n  = glm::vec3(0);
		Particle* pi = sphSystem.particles[i];
		glm::ivec3 cell = sphSystem.getCell(pi);

		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
					uint index = getHash(near_cell);
					Particle* pj = sphSystem.particleTable[index];

					// Iterate through cell linked list
					while (pj != NULL) {
						float dist2 = glm::length2(pj->position - pi->position);
						if (dist2 < sphSystem.H2 && pi != pj) {
							glm::vec3 dir = pj->position - pi->position;
							float W_spikygrad=1000*sphSystem.MASS * (sphSystem.h) / (pj->density) * sphSystem.SPIKY_GRAD;
							n += (dir)*W_spikygrad * (float)pow(sphSystem.h - length(dir), 2);
							//n += normalize(dir) * W_spikygrad;

						}
						pj = pj->next;
					}
				}
			}
		}
		pi->normal = n;
		//if (length2(n) > 0.01 / sphSystem.H2) {
		//	pi->N = 1;
		//	pi->normal = n;
		//	pi->normal_n = normalize(n);
		//}
		//else {
		//	pi->N = 0;
		//	pi->normal = n;
		//	pi->normal_n = glm::vec3(0);
		//}
	}
}
float W_spline(float r, float h) {
	float c = 32 / PI / pow(h, 9);
	if (2 * r > h && r <= h) {
		return c * pow(h - r, 3) * pow(r, 3);
	}
	else if (r > 0 && 2 * r <= h) {
		return c * 2 * pow(h - r, 3) * pow(r, 3) - pow(h, 6) / 64;
	}
	return 0;
}
/**
 * Parallel computation function for calculating forces
 * of particles in the given SPH System.
 */
void parallelForces(const SPHSystem& sphSystem, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle* pi = sphSystem.particles[i];
		pi->force = glm::vec3(0);
		glm::ivec3 cell = sphSystem.getCell(pi);

		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					//find nearby particles in 3x3x3 neighbor
					glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
					uint index = getHash(near_cell);
					Particle* pj = sphSystem.particleTable[index];
					// if there exist a cell in the current neighbor
					// Iterate through cell linked list, update forces applied to pi
					while (pj != NULL) {
						float dist2 = glm::length2(pj->position - pi->position);
						float dist = sqrt(dist2);
						glm::vec3 dir = glm::normalize(pj->position - pi->position);

						if (dist2 < sphSystem.H2 && pi != pj) {
							//unit direction and length

							//apply pressure force
							glm::vec3 pressureForce = -dir * sphSystem.MASS * (pi->pressure + pj->pressure) / (2 * pj->density) * sphSystem.SPIKY_GRAD;
							pressureForce *= std::pow(sphSystem.h - dist, 2);
							pi->force += pressureForce;

							//apply viscosity force
							glm::vec3 velocityDif = pj->velocity - pi->velocity;
							glm::vec3 viscoForce = sphSystem.viscosity * sphSystem.MASS * (velocityDif / pj->density) * sphSystem.SPIKY_LAP * (sphSystem.h - dist);
							pi->force += viscoForce;

							
							//float c = -sphSystem.tension * sphSystem.MASS * ((pj->color - pi->color) / pj->density) * sphSystem.SPIKY_LAP * (sphSystem.h - dist);
							//pi->force += -sphSystem.tension/10 / pi->density * div * pi->normal;
							//if (norm > 0.01 / sphSystem.h) pi->force += c * normalize(n);
						}
					
						if ( pi != pj) {
							//apply surface tension
							//float temp = glm::dot((pj->normal_n - pi->normal_n), dir);
							//float div = (std::min(pi->N, pj->N)) * sphSystem.MASS / pj->density * temp * sphSystem.SPIKY_GRAD;
							//glm::vec3 n = dir * sphSystem.MASS * (pj->color - pi->color) / (pj->density) * sphSystem.SPIKY_GRAD;
							//float norm = length(n);
							
							//cohesion
							float c = W_spline(dist, sphSystem.h);
							float cohesion = -sphSystem.tension* sphSystem.MASS* sphSystem.MASS * c ;
							glm::vec3 cohesionforce = cohesion * dir;

							//pi->force += cohesionforce;

							//curvature
							glm::vec3 curv = -sphSystem.tension * sphSystem.MASS * (pj->normal - pi->normal);

							float k = 2 * sphSystem.restDensity / (pi->density + pj->density);

							glm::vec3 total = k * (cohesionforce + curv);
							if (length(cohesionforce) > 5) {
								k = k;
							}
							pi->force += total;
							pj->force -= total;
						}
						pj = pj->next;
					}
				}
			}
		}
	}
}

/**
 * Parallel computation function moving positions
 * of particles in the given SPH System.
 */
void parallelUpdateParticlePositions(const SPHSystem& sphSystem, float deltaTime, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle *p = sphSystem.particles[i];

		//calculate acceleration and velocity
		glm::vec3 acceleration = p->force / p->density + glm::vec3(0, sphSystem.g, 0);
		p->velocity += acceleration * deltaTime;
		
		// Update position
		p->position += p->velocity * deltaTime;

		// Handle collisions with box
		float boxWidth = sphSystem.bound;
		float elasticity = 0.5f;
		if (p->position.y < p->size) {//if exceeding boudary (no limit at top)
			p->position.y = -p->position.y + 2 * p->size + 0.0001f;
			p->velocity.y = -p->velocity.y * elasticity;// inelastic collision
		}

		if (p->position.x < p->size - boxWidth) {
			p->position.x = -p->position.x + 2 * (p->size - boxWidth) + 0.0001f;
			p->velocity.x = -p->velocity.x * elasticity;
		}

		if (p->position.x > -p->size + boxWidth) {
			p->position.x = -p->position.x + 2 * -(p->size - boxWidth) - 0.0001f;
			p->velocity.x = -p->velocity.x * elasticity;
		}

		if (p->position.z < p->size - boxWidth) {
			p->position.z = -p->position.z + 2 * (p->size - boxWidth) + 0.0001f;
			p->velocity.z = -p->velocity.z * elasticity;
		}

		if (p->position.z > -p->size + boxWidth) {
			p->position.z = -p->position.z + 2 * -(p->size - boxWidth) - 0.0001f;
			p->velocity.z = -p->velocity.z * elasticity;
		}
	}
	//// TODO: after updating particle positions, store current neighboring particles in the vector
	
	//for (int i = start; i < end; i++) {
	//	Particle* pi = sphSystem.particles[i];
	//	glm::ivec3 cell = sphSystem.getCell(pi);

	//	for (int x = -1; x <= 1; x++) {
	//		for (int y = -1; y <= 1; y++) {
	//			for (int z = -1; z <= 1; z++) {
	//				//find nearby particles in 3x3x3 neighbor
	//				glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
	//				uint index = getHash(near_cell);
	//				Particle* pj = sphSystem.particleTable[index];
	//				// if there exist a cell in the current neighbor
	//				// collect and update neighboring particles
	//				while (pj != NULL) {
	//					float dist2 = glm::length2(pj->position - pi->position);
	//					if (dist2 < sphSystem.H2 && pi != pj) {
	//						//std::vector<std::vector<Particle*>> r=sphSystem.neighbouringParticles[0];
	//						
	//					}
	//					pj = pj->next;
	//				}
	//			}
	//		}
	//	}
	//}
	//
	
}
void SPHSystem::updateNeighbor() {
	//// TODO: after updating particle positions, store current neighboring particles in the vector

	for (int i = 0;i<size(particles);i++) {
		Particle* pi = this->particles[i];
		glm::ivec3 cell = this->getCell(pi);
		neighbouringParticles[i].clear();
		int count = 0;
		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					//find nearby particles in 3x3x3 neighbor
					glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
					uint index = getHash(near_cell);
					Particle* pj = this->particleTable[index];
					// if there exist a cell in the current neighbor
					// collect and update neighboring particles
					while (pj != NULL) {
						float dist2 = glm::length2(pj->position - pi->position);
						if (dist2 < this->H2 && pi != pj) {
							//std::vector<std::vector<Particle*>> r=sphSystem.neighbouringParticles[0];
							this->neighbouringParticles[i].push_back(pj);
						}
						pj = pj->next;
					}
				}
			}
		}
	}
}
double SPHSystem::W_poly6(glm::vec3 i, glm::vec3 j, double h) {
	double r2 = length2(i-j);
	double h2 = h * h;
	if (r2 > h2) return 0;
	return POLY6 * pow((h2 - r2), 3);
}
double SPHSystem::W_spikygrad(glm::vec3 i, glm::vec3 j, double h) {
	double r = length(i - j);
	if (r > h) return 0;
	return SPIKY_GRAD * pow(h - r, 2);
}

// TODO: do while exceeding max iteration
// 1. calculate lamda for all particles
// 2. calculate position correction deltaP
// 3. collision detection and response
// 4. update position

void SPHSystem::positionCorrection(float deltatime) {
	int iteration = 0;
	int N = size(particles);
	//std::vector<double>lamda; lamda.resize(N);
	//std::vector<glm::vec3>normal; normal.resize(N);
	while (iteration < MAX_ITERATION) {
		//calcualte lamda
		for (int i = 0; i < N; i++) {
			Particle* pi = particles[i];
			std::vector<Particle*> neighbor = neighbouringParticles[i];
			double density_i = 0;
			double C_i=0;
			double grad_C_i = 0;
			double C_i_pk_sum = 0;
			// case when k=j
			double C_i_pk_j2 = 0;
			glm::vec3 C_i_pk_i=glm::vec3(0);

			double lamda_i;
			for (int j = 0; j < size(neighbor);j++) {
				//double r_2 = length2(pi->position - pj->position);
				Particle* pj = neighbor[j];
				double W = W_poly6(pi->prev, pj->prev, h);
				density_i += W;
				//TODO
				double c = W_spikygrad(pi->prev, pj->prev, h);
				glm::vec3 v = (pi->prev - pj->prev);
				glm::vec3 spikygrad = (float)c * v;
				C_i_pk_i += spikygrad;
				C_i_pk_j2 += length2(spikygrad);


			}
			density_i+= W_poly6(pi->prev, pi->prev, h);
			density_i *= MASS;
			C_i=density_i/restDensity-1;

			C_i_pk_sum += length2(C_i_pk_i);
			C_i_pk_sum /= pow(restDensity, 2);
			lamda_i = -C_i / (C_i_pk_sum+EPS);
			pi->lamda = lamda_i;
		}
		//
		for (int i = 0; i < size(particles); i++) {
			Particle* pi = particles[i];
			std::vector<Particle*> neighbor = neighbouringParticles[i];
			double k = 0.1;
			int n = 4;
			for (int j = 0; j < size(neighbor);j++) {
				Particle *pj = neighbor[j];
				glm::vec3 vir = glm::vec3(0.2*h,0.,0.);
				//debug
				
				double s_corr = -k * pow(W_poly6(pi->prev, pj->prev,h)/W_poly6(vir,glm::vec3(0.,0.,0.),h), n);
				pi->normal +=(float)(pi->lamda + pj->lamda + s_corr) * (float)W_spikygrad(pi->prev, pj->prev, h)
					* (pi->prev - pj->prev);
			}
			pi->normal /= restDensity;
		}
		//update corrected position
		for (int i = 0; i < size(particles); i++) {
			particles[i]->position += particles[i]->normal;
			particles[i]->velocity = (particles[i]->position - particles[i]->prev) / deltatime;
		}
		//
		
		for (int i = 0; i < size(particles); i++) {
			Particle* pi = particles[i];
			std::vector<Particle*> neighbor = neighbouringParticles[i];
			glm::vec3 omega_i=glm::vec3(0.,0.,0.);
			glm::vec3 v_xsph= glm::vec3(0., 0., 0.);
			for (int j = 0; j < size(neighbor); j++) {
				Particle* pj = neighbor[j];
				glm::vec3 pij = pi->prev - pj->prev;
				glm::vec3 vij = -pi->velocity + pj->velocity;
				v_xsph += vij * (float)W_poly6(pij, glm::vec3(.0f,.0f,.0f), h);
				omega_i += glm::cross(vij, (float)W_spikygrad(pij, glm::vec3(0), h) * pij);
			}
			v_xsph *= 0.01;
			pi->velocity += v_xsph;

		}

		iteration++;
	}
}

void SPHSystem::update(float deltaTime) {
	if (!started) return;
	
	// To increase system stability, a fixed deltaTime is set
	deltaTime = 0.003f;

	// Build particle hash table
	buildTable();
	// simulaiton loop first loop
	 // Calculate densities and pressures of particles
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(parallelDensityAndPressures, std::ref(*this), blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

	// Calclulate forces of particles
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(parallelForces, std::ref(*this), blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}


	// store all previous position
	//for (int i = 0; i < size(particles); i++) {
	//	//pre_pos[i] = particles[i]->position;
	//	particles[i]->prev = particles[i]->position;
	//}
	// Update positions of all particles
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(parallelUpdateParticlePositions, std::ref(*this), deltaTime, blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}
}

void SPHSystem::draw(const glm::mat4& viewProjMtx, GLuint shader) {
	// Calculate model matrices for each particle
	for (int i = 0; i < particles.size(); i++) {
		glm::mat4 translate = glm::translate(particles[i]->position);
		sphereModelMtxs[i] = translate * sphereScale;
	}

	// Send matrix data to GPU
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	void* data = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	memcpy(data, sphereModelMtxs, sizeof(glm::mat4) * particles.size());
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Draw instanced particles
	glUseProgram(shader);
	glUniformMatrix4fv(glGetUniformLocation(shader, "viewProjMtx"), 1, false, (float*)&viewProjMtx);
	glBindVertexArray(sphere->vao);
	glDrawElementsInstanced(GL_TRIANGLES, sphere->indices.size(), GL_UNSIGNED_INT, 0, particles.size());
	glBindVertexArray(0);
	glUseProgram(0);
}

/**
 * Parallel helper for clearing table 
 */
void tableClearHelper(SPHSystem& sphSystem, int start, int end) {
	for (int i = start; i < end; i++) {
		sphSystem.particleTable[i] = NULL;
	}
}

/**
 * Parallel helper for building table 
 */
void buildTableHelper(SPHSystem& sphSystem, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle* pi = sphSystem.particles[i];

		// Calculate hash index using hashing formula
		uint index = getHash(sphSystem.getCell(pi));

		// Setup linked list if need be
		mtx.lock();
		if (sphSystem.particleTable[index] == NULL) {
			pi->next = NULL;
			sphSystem.particleTable[index] = pi;
		}
		else {
			pi->next = sphSystem.particleTable[index];
			sphSystem.particleTable[index] = pi;
		}
		mtx.unlock();
	}
}

void SPHSystem::buildTable() {
	// Parallel empty the table
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(tableClearHelper, std::ref(*this), tableBlockBoundaries[i], tableBlockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

	// Parallel build table
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(buildTableHelper, std::ref(*this), blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}
}

glm::ivec3 SPHSystem::getCell(Particle* p) const {
	return glm::ivec3(p->position.x / h, p->position.y / h, p->position.z / h);
}

void SPHSystem::reset() {
	initParticles();
	started = false;
}

void SPHSystem::startSimulation() {
	started = true;
}
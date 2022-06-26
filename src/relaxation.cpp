#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <mpi.h>

const double guess = 0.0;
const double V_left = 5.0;
const double V_right = 1.0;
const size_t SIZE = 100;
const size_t iterations = 10000;

std::vector<double> lattice(SIZE, guess);
std::vector<double> ghost_values;

void update_lattice(const int& start, const int& end, const int task_id) {
    lattice[start] = (1.0 / 2.0) * (ghost_values[task_id * 2] + lattice.at(start + 1));
    for (int i = start + 1; i < end - 1; i++) {
	lattice[i] = (1.0 / 2.0) * (lattice[i + 1] + lattice[i - 1]);	
    }
    lattice[end - 1] = (1.0 / 2.0) * (lattice[end - 2] + ghost_values[(2 * task_id) + 1]); 
}
void update_ghosts(const int chunk_size, const int leftover) {
    int guide = chunk_size + leftover + 1;
    ghost_values[1] = (1.0 / 2.0) * (lattice.at(guide + 1) + lattice.at(guide - 1));
    guide -= 1;
    for (int i = 2; i < ghost_values.size() - 1; i++) {
	ghost_values[i] = (1.0 / 2.0) * (lattice.at(guide + 1) + lattice.at(guide - 1));
        guide = i % 2 == 0 ? guide + chunk_size + 1 : guide - 1;
    }
}

void print() {
    std::ofstream out;    
    out.open("out.txt"); 
    for (int i = 0; i < SIZE; i++){ 
	out << (double)i / (double)(SIZE - 1) << " "  
                  << lattice[i] << "\n";
    }
}

int main(int argc, char **argv){ 
    lattice[0] = V_left;
    lattice[SIZE - 1] = V_right;
    int start;
    int num_tasks;
    int task_id;	
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
	
    int chunk_size = (SIZE - 2) / num_tasks; 
    int leftover = (SIZE - 2) % num_tasks;	
    ghost_values.resize(2 * num_tasks, guess);
    ghost_values[0] = V_left;
    ghost_values[ghost_values.size() - 1] = V_right; 
	
    for (int i = 0; i <= iterations; i++){ 
        if (task_id == 0) {
	    if (i == iterations) {
	        print();
		break;
	    }
	    start = chunk_size + leftover + 1;  
	    for (int dest = 1; dest < num_tasks; dest++) { 
		MPI_Send(&start, 1, MPI_INT, dest, 1, MPI_COMM_WORLD); 
		MPI_Send(&lattice[start], chunk_size, MPI_DOUBLE, dest, 2, MPI_COMM_WORLD);
		MPI_Send(&ghost_values[dest * 2], 2, MPI_DOUBLE, dest, 3, MPI_COMM_WORLD);
		start += chunk_size;
	    }

	    start = 1;
	    update_lattice(start, chunk_size + leftover + 1, 0);

	    for (int source = 1; source < num_tasks; source++) {
		MPI_Recv(&start, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&lattice[start], chunk_size, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
		MPI_Recv(&ghost_values[source * 2], 2, MPI_DOUBLE, source, 3, MPI_COMM_WORLD, &status);
	    }

	    update_ghosts(chunk_size, leftover);
	}
	else if (task_id > 0) {
	    if (i == iterations) {
	        break;
	    }
	    MPI_Recv(&start, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
	    MPI_Recv(&lattice[start], chunk_size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);
	    MPI_Recv(&ghost_values[task_id * 2], 2, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &status);

	    update_lattice(start, chunk_size + start, task_id);

	    MPI_Send(&start, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
	    MPI_Send(&lattice[start], chunk_size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
	    MPI_Send(&ghost_values[task_id * 2], 2, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
	}
    }
    MPI_Finalize();

    return 0;
}

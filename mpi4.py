
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt


N_POINTS = 481
DOMAIN_SIZE = 1.0
N_ITERATIONS = 3000
TIME_STEP_LENGTH = 0.00001
KINEMATIC_VISCOSITY = 0.1
DENSITY = 1.0
HORIZONTAL_VELOCITY_TOP = 2.0

N_PRESSURE_POISSON_ITERATIONS = 50
STABILITY_SAFETY_FACTOR = 0.5

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running on {size} processes")

    element_length = DOMAIN_SIZE / (N_POINTS - 1)
    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)

    local_N_POINTS = N_POINTS // size + 2
    local_domain_size = float(local_N_POINTS - 2) / (N_POINTS - 1)

    y = np.linspace(rank * local_domain_size, (rank + 1) * local_domain_size, local_N_POINTS)
    X, Y = np.meshgrid(x, y)

    u_prev = np.zeros_like(X)
    v_prev = np.zeros_like(X)
    p_prev = np.zeros_like(X)

    def central_difference_x(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, 2:  ]
            -
            f[1:-1, 0:-2]
        ) / (
            2 * element_length
        )
        return diff
    
    def central_difference_y(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[2:  , 1:-1]
            -
            f[0:-2, 1:-1]
        ) / (
            2 * element_length
        )
        return diff
    
    def laplace(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, 0:-2]
            +
            f[0:-2, 1:-1]
            -
            4
            *
            f[1:-1, 1:-1]
            +
            f[1:-1, 2:  ]
            +
            f[2:  , 1:-1]
        ) / (
            element_length**2
        )
        return diff

    maximum_possible_time_step_length = (
        0.5 * element_length**2 / KINEMATIC_VISCOSITY
    )
    if TIME_STEP_LENGTH > STABILITY_SAFETY_FACTOR * maximum_possible_time_step_length:
        raise RuntimeError("Stability is not guarenteed")

    for iteration in range(N_ITERATIONS):
        d_u_prev__d_x = central_difference_x(u_prev)
        d_u_prev__d_y = central_difference_y(u_prev)
        d_v_prev__d_x = central_difference_x(v_prev)
        d_v_prev__d_y = central_difference_y(v_prev)
        laplace__u_prev = laplace(u_prev)
        laplace__v_prev = laplace(v_prev)

        u_tent = (
            u_prev
            +
            TIME_STEP_LENGTH * (
                -
                (
                    u_prev * d_u_prev__d_x
                    +
                    v_prev * d_u_prev__d_y
                )
                +
                KINEMATIC_VISCOSITY * laplace__u_prev
            )
        )

        v_tent = (
            v_prev
            +
            TIME_STEP_LENGTH * (
                -
                (
                    u_prev * d_v_prev__d_x
                    +
                    v_prev * d_v_prev__d_y
                )
                +
                KINEMATIC_VISCOSITY * laplace__v_prev
            )
        )

        u_tent[0, :] = 0.0
        u_tent[:, 0] = 0.0
        u_tent[:, -1] = 0.0
        u_tent[-1, :] = HORIZONTAL_VELOCITY_TOP
        v_tent[0, :] = 0.0
        v_tent[:, 0] = 0.0
        v_tent[:, -1] = 0.0
        v_tent[-1, :] = 0.0

        div_u_tent = central_difference_x(u_tent) + central_difference_y(v_tent)
        rhs = DENSITY / TIME_STEP_LENGTH * div_u_tent
        p_next = np.zeros_like(p_prev)
        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            p_next[1:-1, 1:-1] = 1/4 * (
                p_prev[1:-1, 0:-2]
                + p_prev[0:-2, 1:-1]
                + p_prev[1:-1, 2:  ]
                + p_prev[2:  , 1:-1]
                - element_length**2 * rhs[1:-1, 1:-1]
            )

            if rank == 0:
                p_next[0, :] = p_next[1, :]
            if rank == size - 1:
                p_next[-1, :] = 0.0

            p_next[:,  0] = p_next[:,  1]
            p_next[:, -1] = p_next[:, -2]

            comm.Barrier()
            
            if rank > 0:
                send_buffer = np.array(p_next[1, :])
                recv_buffer = np.empty(p_next[1, :].shape)
                comm.Sendrecv(send_buffer, dest=rank - 1, recvbuf=recv_buffer, source=rank - 1)
                p_next[0, :] = recv_buffer
            if rank < size - 1:
                send_buffer = np.array(p_next[-2, :])
                recv_buffer = np.empty(p_next[-2, :].shape)
                comm.Sendrecv(send_buffer, dest=rank + 1, recvbuf=recv_buffer, source=rank + 1)
                p_next[-1, :] = recv_buffer

            if rank < size - 1:
                send_buffer = np.array(p_next[-3, :])
                recv_buffer = np.empty(p_next[-3, :].shape)
                comm.Sendrecv(send_buffer, dest=rank + 1, recvbuf=recv_buffer, source=rank + 1)
                p_next[-2, :] = recv_buffer
            if rank > 0:
                send_buffer = np.array(p_next[2, :])
                recv_buffer = np.empty(p_next[2, :].shape)
                comm.Sendrecv(send_buffer, dest=rank - 1, recvbuf=recv_buffer, source=rank - 1)
                p_next[1, :] = recv_buffer

            comm.Barrier()

            p_prev = p_next

        u_next = u_tent - TIME_STEP_LENGTH / DENSITY * central_difference_x(p_next)
        v_next = v_tent - TIME_STEP_LENGTH / DENSITY * central_difference_y(p_next)

        u_next[0, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0
        u_next[-1, :] = HORIZONTAL_VELOCITY_TOP
        v_next[0, :] = 0.0
        v_next[:, 0] = 0.0
        v_next[:, -1] = 0.0
        v_next[-1, :] = 0.0

        u_prev = u_next
        v_prev = v_next
        p_prev = p_next

    pressures = comm.gather(p_next[1:-1, :], root=0)
    velocities_u = comm.gather(u_next[1:-1, :], root=0)
    velocities_v = comm.gather(v_next[1:-1, :], root=0)

    if rank == 0:
        p_all = np.vstack(pressures)
        u_all = np.vstack(velocities_u)
        v_all = np.vstack(velocities_v)
        X_full, Y_full = np.meshgrid(np.linspace(0.0, DOMAIN_SIZE, p_all.shape[1]), np.linspace(0.0, DOMAIN_SIZE, p_all.shape[0]))
        plt.style.use("dark_background")
        plt.figure()
        plt.contourf(X_full[::2, ::2], Y_full[::2, ::2], p_all[::2, ::2], cmap="coolwarm")
        plt.colorbar()
        plt.quiver(X_full[::2, ::2], Y_full[::2, ::2], u_all[::2, ::2], v_all[::2, ::2], color="black")
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.savefig('Lid_Driven_Cavity.png')

if __name__ == "__main__":
    main()



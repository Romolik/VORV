#include <stdio.h>
#include <mpi.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <queue>
#include <set>
#include <algorithm>

//https://neerc.ifmo.ru/wiki/index.php?title=Алгоритм_Лампорта_взаимного_исключения

#define REQ 11
#define OK 12
#define RELEASE 13

typedef struct block {
    int rank;
    long timestamp;
    int num_received_ok;
} block_t;

std::vector<block_t> blocks;

int mpi_rank, mpi_size;
long timestamp = 0;
long max_timestamp = 0;
block_t* my_request;

auto cmp = [](block_t a, block_t b) {

    if (a.timestamp == b.timestamp) {
        return a.rank < b.rank;
    } else {
        return a.timestamp < b.timestamp;
    }
};

std::set<block_t, decltype(cmp)> request_queue;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;


void queue_add(std::set<block_t, decltype(cmp)> *queue, block_t* block) {

    queue->insert(*block);
    block->num_received_ok = 0;
}

void queue_remove(std::set<block_t, decltype(cmp)> *queue, block* block) {

    queue->erase(*block);
}

void send_message_to_other_processes(int tag, long msg) {

    MPI_Request requests[mpi_size];

    for (int i = 0; i < mpi_size; i++) {
        if (i == mpi_rank) {
            continue;
        }

        MPI_Isend(&msg, 1, MPI_LONG, i, tag, MPI_COMM_WORLD, &requests[i]);
    }
}

void *receiver(void *param) {

    while (true) {
        long recv_timestamp;
        MPI_Status status;
        MPI_Recv(&recv_timestamp, 1, MPI_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;

        if (source == mpi_rank) {
            continue;
        }

        max_timestamp = std::max(max_timestamp, recv_timestamp);

        //Если другой процесс присылает RELEASE, то удаляем его запрос из очереди
        if (tag == RELEASE) {
            pthread_mutex_lock(&mutex);

            queue_remove(&request_queue, &blocks[source]);

            pthread_mutex_unlock(&mutex);
            pthread_cond_signal(&cond);
        } else if (tag == REQ) {
            block_t *block = &blocks[source];
            block->timestamp = recv_timestamp;

            pthread_mutex_lock(&mutex);
            queue_add(&request_queue, block);
            pthread_mutex_unlock(&mutex);

            clock_t msg = timestamp;
            MPI_Send(&msg, 1, MPI_LONG, source, OK, MPI_COMM_WORLD);
        } else if (tag == OK) {
            //Увеличиваем счётчик полученных ответов ОК, если мы получили все ОК, то вызываем сигнал
            int cond_signal = false;

            pthread_mutex_lock(&mutex);

            my_request->num_received_ok++;

            if (my_request->num_received_ok >= mpi_size - 1) {
                cond_signal = true;
            }

            pthread_mutex_unlock(&mutex);

            if (cond_signal) {
                pthread_cond_signal(&cond);
            }
        } else {
            fprintf(stderr, "Получен не существующий статус\n");
        }
    }
}

bool isEquals(auto a, block_t *b) {

    return a->rank == b->rank && a->timestamp == b->timestamp;
}

void init() {

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    blocks.reserve(mpi_size);

    for (int i = 0; i < mpi_size; i++) {
        block_t block;
        block.rank = i;
        blocks[i] = block;
    }

    my_request = &blocks[mpi_rank];
    my_request->num_received_ok = 0;
    timestamp = 0;
}

void lock() {

    pthread_mutex_lock(&mutex);

    my_request->timestamp = ++timestamp;
    //Добавляем свой запрос в свою очередь (временную метку и номер процесса)
    queue_add(&request_queue, my_request);

    long msg = timestamp;
    //Отправляем всем процессам req запрос с нашим текущим временем
    send_message_to_other_processes(REQ, msg);
    my_request->num_received_ok = 0;

    //Ждём, когда все процессы вернут нам ОК и наш request станет первым в очереди
    while (!isEquals(request_queue.begin(), my_request) || my_request->num_received_ok < mpi_size - 1) {
        pthread_cond_wait(&cond, &mutex);
    }

    pthread_mutex_unlock(&mutex);
}

void unlock() {

    long msg = timestamp;

    pthread_mutex_lock(&mutex);

    queue_remove(&request_queue, my_request);
    timestamp = std::max(timestamp, max_timestamp + 1);

    pthread_mutex_unlock(&mutex);

    send_message_to_other_processes(RELEASE, msg);
}

int main(int argc, char **argv) {

    int required = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(&argc, &argv, required, &provided);

    if (required != provided) {
        fprintf(stderr, "MPI_Init_thread: required %d, provided %d\n", required, provided);
    }

    init();

    pthread_t recv_thread;
    int errCode = pthread_create(&recv_thread, NULL, receiver, NULL);
    if (errCode != 0) {
        fprintf(stderr, "Unable to create thread: %s\n", strerror(errCode));
        MPI_Finalize();
        return 0;
    }

    while (true) {
        lock();
        printf("Message from %d\n", mpi_rank);
        usleep(5000);
        unlock();
    }

    MPI_Finalize();
    return 0;
}
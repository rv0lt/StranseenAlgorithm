RM = /bin/rm -f
CC = gcc
CFLAGS = -Wall -O3 -fopenmp -ffast-math
LDFLAGS= -lm -fopenmp

OBJS = strassen.o
EXECUTABLE = strassen 

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)
strassen.o: strassen.c
	$(CC) $(CFLAGS) -c strassen.c
clean:
	$(RM) $(EXECUTABLE) $(OBJS)



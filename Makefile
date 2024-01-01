TARGET_EXEC := mnist_nn

SRC_DIR := src
BUILD_DIR := build

SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%,$(BUILD_DIR)/%,$(SRCS:.c=.o))

CC = gcc
CFLAGS = -std=c17 -I include -Wall -lm
LDFLAGS = -lm

TARGET := mnist_nn

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@


.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

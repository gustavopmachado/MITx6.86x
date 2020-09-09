def get_sum_metrics(predictions, metrics=None):

    # An empty list should never be used as default argument, since the default arguments are only
    # evaluated when the 'def' command - which initiates the function object - is executed. Therefore, it's
    # evaluated only once. Mutable objects, such as lists, shouldn't be used as default arguments
    if metrics is None:
        metrics = []

    # The functions created point to the variable i, which will be 2 at the end of the loop. Therefore, every
    # function will point to the same variable i, which will be the last result. In order to avoid that, it's
    # necessary to create a function:
    def myFunc(i):
        return lambda x: x + i

    for i in range(3):
        metrics.append(myFunc(i))

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)

    return sum_metrics


def main():
    print(get_sum_metrics(0))  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
    print(get_sum_metrics(1))  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
    print(get_sum_metrics(2))  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9
    # Should be (3) + (3 + 0) + (3 + 1) + (3 + 2) = 15
    print(get_sum_metrics(3, [lambda x: x]))
    print(get_sum_metrics(0))  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
    print(get_sum_metrics(1))  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
    print(get_sum_metrics(2))  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9


if __name__ == "__main__":
    main()

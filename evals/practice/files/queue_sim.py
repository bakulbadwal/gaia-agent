"""Ticket-counter simulation used for weekend capacity planning."""


def arrivals(minute):
    """Customers arriving in a given minute: a rush at the top and bottom of each hour."""
    if minute % 30 < 5:
        return 3
    return 1 if minute % 3 == 0 else 0


def simulate(minutes, counters=2, service_time=4):
    """Return how many customers are still queueing (not being served) after `minutes`."""
    queue = 0
    busy = [0] * counters  # minutes left on the customer at each counter
    for t in range(minutes):
        queue += arrivals(t)
        for i in range(counters):
            if busy[i] > 0:
                busy[i] -= 1
            if busy[i] == 0 and queue > 0:
                queue -= 1
                busy[i] = service_time
    return queue


if __name__ == "__main__":
    print("customers waiting after 60 minutes:", simulate(60))

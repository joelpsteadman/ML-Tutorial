import random

class Data_Set:
    def __init__(self, 
                x_min = 0,
                x_max = 100,
                y_min = 0,
                y_max = 100) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.y_min= y_min
        self.y_max = y_max

    def generate_delimiter(self):
        self.intercept = random.uniform(self.y_min, self.y_max)

        x_range = self.x_max - self.x_min
        min_slope = (self.y_min - self.intercept) / x_range
        max_slope = (self.y_max - self.intercept) / x_range
        # Creates a slope that will not make the line go outside of the graph
        self.slope = (random.uniform(min_slope, max_slope))

    def separation_function(self, x):
        return self.slope * x + self.intercept

    def desired(self, x, y):
        if y > self.separation_function(x):
            return 1
        else:
            return 0

    def generate_training_set(self, size):
        x_points = []
        y_points = []
        for i in range(size):
            x_points.append(random.uniform(self.x_min, self.x_max))
            y_points.append(random.uniform(self.y_min, self.y_max))
        return [x_points, y_points]

    def generate_testing_set(self, size):
        x_test_cases = []
        y_test_cases = []
        for i in range(size):
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            x_test_cases.append(x)
            y_test_cases.append(y)
        return [x_test_cases, y_test_cases]

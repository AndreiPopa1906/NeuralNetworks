import re


def parse_equations(file_path: str) -> list[list]:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    A = []  # Coefficients
    B = []  # Results

    first_line = lines[0]
    variable_matches = re.findall(r'[a-zA-Z]+', first_line)
    variables = list(dict.fromkeys(variable_matches))
    # print("Variables: ", variables)

    for line in lines:
        matches = []
        for var in variables:
            match = re.search(r'([+-]?\d*)' + var, line)
            if match:
                matches.append(match.group(1))
            else:
                matches.append('0')
        constant = re.search(r'=\s*([+-]?\d+)', line)

        coefficients = []
        for m in matches:
            if m not in ('', '+', '-'):
                coefficients.append(int(m))
            else:
                coefficients.append(1 if m != '-' else -1)

        A.append(coefficients)

        if constant:
            B.append(int(constant.group(1)))

    return A, B


class Matrix():
    def __init__(self, values: list[list]):
        if isinstance(values, list):
            self.values = values
            self.n = len(values)
            self.m = len(values[0])
        elif isinstance(values, tuple) and len(values) == 2:
            self.n, self.m = values
            self.values = [[0 for _ in range(self.m)] for _ in range(self.n)]
        else:
            raise ValueError("Invalid constructor arguments for Matrix class")

    def get_minor(self, row: int, column: int) -> 'Matrix':
        minor_values = [r[:column] + r[column + 1:] for i, r in enumerate(self.values) if i != row]
        return Matrix(minor_values)

    def get_determinant(self) -> int:
        if self.n != self.m:
            return "Matrix is not square"

        if self.n == 2:
            return self.values[0][0] * self.values[1][1] - self.values[0][1] * self.values[1][0]

        determinant = 0
        for j in range(self.n):
            minor = self.get_minor(0, j)
            cofactor = ((-1) ** j) * self.values[0][j] * minor.get_determinant()
            determinant += cofactor

        return determinant

    def get_trace(self) -> int:
        if self.n != self.m:
            return "Matrix is not square"

        trace = 0
        for i in range(self.n):
            trace += self.values[i][i]

        return trace

    def get_vector_norm(self) -> int:
        if self.m != 1:
            return "Matrix is not a vector"

        norm = 0
        for i in range(self.n):
            norm += self.values[i][0] ** 2

        return norm ** 0.5

    def get_transpose(self) -> 'Matrix':
        transpose_values = [[self.values[j][i] for j in range(self.n)] for i in range(self.m)]
        return Matrix(transpose_values)

    def multiply_vector(self, vector: list) -> list:
        if self.m != len(vector):
            return "Matrix and vector dimensions do not match"

        result = []
        for i in range(self.n):
            row = self.values[i]
            result.append(sum([row[j] * vector[j] for j in range(self.m)]))

        return result

    def replace_column(self, column_index: int, new_column: int) -> 'Matrix':
        modified_values = [row[:] for row in self.values]
        for i in range(self.n):
            modified_values[i][column_index] = new_column[i]
        return Matrix(modified_values)

    def get_inverse(self) -> 'Matrix':
        if self.n != self.m:
            return "Matrix is not square"

        determinant = self.get_determinant()
        if determinant == 0:
            return "Matrix is not invertible"

        cofactor_matrix = Matrix((self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                minor = self.get_minor(i, j)
                cofactor_matrix.values[i][j] = (-1) ** (i + j) * minor.get_determinant()

        adjugate_matrix = cofactor_matrix.get_transpose()

        inverse_values = [[adjugate_matrix.values[i][j] / determinant for j in range(self.m)] for i in range(self.n)]

        return Matrix(inverse_values)

    def solve_using_inverse(self, vector: int) -> list:
        if len(vector) != self.n:
            return "Vector dimensions do not match matrix dimensions"

        inverse_matrix = self.get_inverse()
        solution = inverse_matrix.multiply_vector(vector)

        return solution


if __name__ == "__main__":
    file_path = 'input.txt'
    A_values, B = parse_equations(file_path)
    print("Matrix A (Coefficients):")
    for r in A_values:
        print(r)
    print("Vector B (Constants/Results):")
    print(B)

    A = Matrix(A_values)
    det_A = A.get_determinant()

    if det_A == 0:
        print("The system does not have a unique solution.")
    else:
        results = []
        for i in range(len(B)):
            A_modified = A.replace_column(i, B)
            det_A_modified = A_modified.get_determinant()
            results.append(det_A_modified / det_A)

        print("Solution:")
        for idx, value in enumerate(results):
            print(f"x{idx + 1} = {value}")

        try:
            solution_inverse = A.solve_using_inverse(B)
            print("Solution using Inverse Method:")
            for idx, value in enumerate(solution_inverse):
                print(f"x{idx + 1} = {value}")
        except ValueError as e:
            print(e)

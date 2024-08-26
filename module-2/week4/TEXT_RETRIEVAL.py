import math


class Solution:
    def calculate_eigenvalues_eigenvectors(self, A):
        def get_eigenvalues(matrix):
            a, b = matrix[0][0], matrix[0][1]
            c, d = matrix[1][0], matrix[1][1]

            trace = a + d
            determinant = a * d - b * c
            discriminant = trace**2 - 4 * determinant

            if discriminant < 0:
                raise ValueError(
                    "Eigenvalues are complex and cannot be computed with the math library.")

            sqrt_discriminant = math.sqrt(discriminant)
            eigenvalue1 = (trace + sqrt_discriminant) / 2
            eigenvalue2 = (trace - sqrt_discriminant) / 2

            return eigenvalue1, eigenvalue2

        def get_eigenvector(matrix, eigenvalue):
            a, b = matrix[0][0], matrix[0][1]
            c, d = matrix[1][0], matrix[1][1]

            if b != 0:
                eigenvector = [(eigenvalue - d) / b, 1]
            else:
                eigenvector = [1, (eigenvalue - a) / c]

            norm = math.sqrt(eigenvector[0]**2 + eigenvector[1]**2)
            eigenvector = [x / norm for x in eigenvector]

            return eigenvector

        eigenvalue1, eigenvalue2 = get_eigenvalues(A)
        eigenvector1 = get_eigenvector(A, eigenvalue1)
        eigenvector2 = get_eigenvector(A, eigenvalue2)

        # Create dictionary with eigenvalues as keys and eigenvectors as values
        return {eigenvalue1: eigenvector1, eigenvalue2: eigenvector2}


# Example usage:
sol = Solution()
A = [[2, 1], [1, 2]]
print(sol.calculate_eigenvalues_eigenvectors(A))

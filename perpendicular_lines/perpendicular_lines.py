import math

def perpendicular_or_not(x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, x4: int, y4: int) -> str:
    """
    :param x1: X-axis for first coordinate
    :param y1: Y-axis for first coordinate
    :param x2: X-axis for second coordinate
    :param y2: Y-axis for second coordinate
    :param x3: X-axis for third coordinate
    :param y3: Y-axis for third coordinate
    :param x4: X-axis for fourth coordinate
    :param y4: Y-axis for fourth coordinate
    :return: Return "YES" if lines are perpendicular, "NO" if not and "INVALID" if points are equal or those lines are collinear.
    """

    if (x2 == x1 and y2 == y1) or (x4 == x3 and y4 == y3):
        return "INVALID"

    if x2 == x1:
        m1 = math.inf
        c1 = x1
    else:
        m1 = (y2 - y1)/(x2 - x1)
        c1 = y2 - (m1 * x2)

    if x4 == x3:
        m2 = math.inf
        c2 = x3
    else:
        m2 = (y4 - y3)/(x4 - x3)
        c2 = y4 - (m2 * x4)

    m1_cross_m2 = m1 * m2

    if m1_cross_m2 == -1:
        return "YES"
    elif m1 == m2 and c1 == c2:
        return "INVALID"
    else:
        return "NO"


if __name__ == "__main__":
    test_cases = int(input())
    while test_cases > 0:
        first_two_points = input()
        x1, y1, x2, y2 = [int(x) for x in first_two_points.split()]
        second_two_points = input()
        x3, y3, x4, y4 = [int(x) for x in second_two_points.split()]
        print(perpendicular_or_not(x1, y1, x2, y2, x3, y3, x4, y4))
        test_cases -= 1

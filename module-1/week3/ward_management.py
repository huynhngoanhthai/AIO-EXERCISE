class Person:
    def __init__(self, name: str, yob: int) -> None:
        self.name = name
        self.yob = yob

    def describe(self) -> None:
        # pass
        pass


class Student(Person):
    def __init__(self, name: str, yob: int, grade: str) -> None:
        super().__init__(name, yob)
        self.grade = grade

    def describe(self) -> None:
        print(
            f"Student - Name: {self.name} - YoB: {self.yob} - Grade: {self.grade}")


class Teacher(Person):
    def __init__(self, name: str, yob: int, subject: str) -> None:
        super().__init__(name, yob)
        self.subject = subject

    def describe(self) -> None:
        print(
            f"Teacher - Name: {self.name} - YoB: {self.yob} - Subject: {self.subject}")


class Doctor(Person):
    def __init__(self, name: str, yob: int, specialist: str) -> None:
        super().__init__(name, yob)
        self.specialist = specialist

    def describe(self) -> None:
        print(
            f"Doctor - Name: {self.name} - YoB: {self.yob} - Specialist: {self.specialist}")


class Ward:
    def __init__(self, name: str) -> None:
        self.name = name
        self.people = []

    def add_person(self, person: Person) -> None:
        self.people.append(person)

    def describe(self) -> None:
        print(f"Ward Name: {self.name}")
        for person in self.people:
            person.describe()

    def count_doctor(self) -> int:
        return sum(1 for person in self.people if isinstance(person, Doctor))

    def sort_age(self) -> None:
        self.people.sort(key=lambda person: person.yob)

    def compute_average(self) -> float:
        teachers = [
            person for person in self.people if isinstance(person, Teacher)]
        if not teachers:
            return 0.0
        return sum(teacher.yob for teacher in teachers) / len(teachers)


student1 = Student(name="studentA", yob=2010, grade="7")
student1.describe()

teacher1 = Teacher(name="teacherA", yob=1969, subject="Math")
teacher1.describe()

doctor1 = Doctor(name="doctorA", yob=1945, specialist="Endocrinologists")
doctor1.describe()

teacher2 = Teacher(name="teacherB", yob=1995, subject="History")
doctor2 = Doctor(name="doctorB", yob=1975, specialist="Cardiologists")
ward1 = Ward(name="Ward1")
ward1.add_person(student1)
ward1.add_person(teacher1)
ward1.add_person(teacher2)
ward1.add_person(doctor1)
ward1.add_person(doctor2)
ward1.describe()


print(f"Number of doctors: {ward1.count_doctor()}")
ward1.sort_age()
ward1.describe()

average_yob = ward1.compute_average()
print(f"Average year of birth of teachers: {average_yob}")

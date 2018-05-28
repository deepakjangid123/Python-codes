# Parent class
class Dog:

    # Class attribute
    species = 'mammel'

    # Initializer / Instance attributes
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Instance method
    def description(self):
        return "{} is {} years old".format(self.name, self.age)

    # Instance method
    def speak(self, sound):
        return "{} says {}".format(self.name, sound)


# Child class (inherits from Dog() class)
class BullDog(Dog):
    def run(self, speed):
        return "{} runs {}".format(self.name, speed)


# Child class (inherits from Dog() class)
class RussellTerrier(Dog):
    def run(self, speed):
        return "{} runs {}".format(self.name, speed)

if __name__ == '__main__':
    # Child classes inherit attributes and
    # behaviors from the parent class
    jim = BullDog("Jim", 12)
    print(jim.description())

    # Child classes have specific attributes
    # and behaviors as well
    print(jim.run("slowly"))

    # Is jim an instance of Dog()?
    print(isinstance(jim, Dog))

    # Is julie an instance of Dog()?
    julie = Dog("Julie", 100)
    print(isinstance(julie, Dog))

    # Is johnny walker an instance of Bulldog()
    johnnywalker = RussellTerrier("Johnny Walker", 4)
    print(isinstance(johnnywalker, RussellTerrier))
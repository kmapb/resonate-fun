from resonate import Resonate, Context
import random


# Initialize Resonate with a local store
resonate = Resonate()


# Register add() with Resonate
@resonate.register
def add(_: Context, a: int, b: int):
    return a + b


def main():
    try:
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        print(f"adding {a} and {b}")
        handle = add.run(f"add.{a}+{b}", a, b)
        print(f"result: {handle.result()}")
    except Exception as e:
        print({"error": str(e)})


if __name__ == "__main__":
    main()

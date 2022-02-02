import utils
import sys
from labyrinth_env import LabyrinthEnv
from q_learning import QLearning

if __name__ == "__main__":

    print(
        "*******************************\n"
        "\tLabyrinth Agent\n"
        "*******************************\n"
    )

    lab_input = input(
        "Do you want to load a saved Labyrinth?\n"
        "(1)\tYes\n"
        "(2)\tNo, generate one\n"
        "(0)\tExit\n"
    )

    env = None

    if lab_input == "1":
        max_actions = int(input("Max actions: "))
        env = LabyrinthEnv(max_actions=max_actions, load=True)

    elif lab_input == "2":
        grid_h = int(input("Grid height: "))
        grid_w = int(input("Grid Width: "))
        wall_percentage = int(input("Wall percentage:"))
        max_actions = int(input("Max actions: "))
        env = LabyrinthEnv(
            max_actions=max_actions,
            grid_h=grid_h,
            grid_w=grid_w,
            wall_percentage=wall_percentage,
        )

    else:
        print("End\n")
        sys.exit()

    QL = QLearning(env)
    utils.clear_screen()

    env.render()
    menu_input = input(
        "(1)\tTraining\n"
        "(2)\tExecute automatically\n"
        "(3)\tExecute step-by-step\n"
        "(0)\tExit\n"
    )
    utils.clear_screen()

    if menu_input == "1":
        print("Training\n")
        QL.training(epochs=25000, steps=400, alpha=0.1, gamma=1.0, eps=1.0, plot=True)

    elif menu_input == "2":
        print("Execute\n")
        QL.execute(step_by_step=False)

    elif menu_input == "3":
        print("Execute\n")
        QL.execute(step_by_step=True)

    else:
        print("End\n")

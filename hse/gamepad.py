import pygame


class Gamepad:
    """
    A class to interact with a gamepad.
    """

    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count < 1:
            raise RuntimeError("Joystick not found")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def print_info(self):
        print(f"Joystick Name: {self.joystick.get_name()}")
        print(f"Number of Axes: {self.joystick.get_numaxes()}")
        print(f"Number of Buttons: {self.joystick.get_numbuttons()}")

    # def get_state(self):
    #     joystick_state = {}
    #     for event in pygame.event.get():
    #         if event.type == pygame.JOYAXISMOTION:
    #             axis_idx = event.axis
    #             axis_val = event.value  # -1.0 ~ 1.0
    #             joystick_state[f"axis_{axis_idx}"] = axis_val
    #     return joystick_state

    def get_state(self):
        """
        Returns the current state of the given joystick as a dictionary.
        """
        pygame.event.pump()  # Update internal state
        joystick = self.joystick

        # state = { # dict
        #     "axes": {
        #         f"axis_{i}": joystick.get_axis(i) for i in range(joystick.get_numaxes())
        #     },
        #     "buttons": {
        #         f"button_{i}": joystick.get_button(i)
        #         for i in range(joystick.get_numbuttons())
        #     },
        #     "hats": {
        #         f"hat_{i}": joystick.get_hat(i) for i in range(joystick.get_numhats())
        #     },
        # }
        state = {  # list
            "axes": [joystick.get_axis(i) for i in range(joystick.get_numaxes())],
            "buttons": [
                joystick.get_button(i) for i in range(joystick.get_numbuttons())
            ],
            "hats": [joystick.get_hat(i) for i in range(joystick.get_numhats())],
        }
        return state

    def test_gamepad(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.JOYAXISMOTION:
                    axis_idx = event.axis
                    axis_val = event.value  # -1.0 ~ 1.0

                    print(f"Axis {axis_idx} moved to {axis_val:.3f}")

    # pygame.quit()


if __name__ == "__main__":
    gamepad = Gamepad()
    gamepad.print_info()
    gamepad.test_gamepad()

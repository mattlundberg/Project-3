import reflex as rx
import random
from rxconfig import config
from reflex_type_animation import type_animation

class State(rx.State):
    output_count: int = 0
    post_text: str = ""
    output_label: str = "74% False"
    submit_phrase: str | None = None
    animation_key: int = 0

    submit_phrases = [
        "Feeding Fake News Hamsters...",
        "Combing Through Dumpster X...",
        "Greasing the Gears of Progress...",
        "Polishing the Data Diamonds...",
        "Waking Up the Server Gremlins...",
        "Tickling the Code Monkeys...",
        "Charging the Quantum Batteries...",
        "Brewing the Algorithm Soup...",
        "Herding Digital Cats...",
        "Spinning the Data Wheel...",
        "Sharpening the Logic Knives...",
        "Tuning the Neural Networks...",
        "Stirring the Binary Cauldron...",
        "Training the Robot Ninjas...",
        "Dusting Off the Server Cobwebs...",
        "Juggling Bits and Bytes...",
        "Chasing Down Bug Gremlins...",
        "Wiring the Synapse Circuits...",
        "Squeezing the Data Lemons...",
        "Summoning the Code Wizards..."
    ]

    @rx.var
    def output_count_label(self) -> str:
        return f"Count >75%: {self.output_count}"

    def set_post_text(self, value):
        self.post_text = value

    def generate_post(self):
        fake_posts = [
            "BREAKING: Scientists discover chocolate cures all diseases!",
            "Aliens land in Central Park, demand pizza.",
            "New study shows cats secretly control the government."
        ]
        real_posts = [
            "NASA successfully launches new Mars rover.",
            "WHO recommends regular hand washing to prevent illness.",
            "Local library hosts summer reading program for kids."
        ]
        if random.choice([True, False]):
            post = random.choice(fake_posts)
        else:
            post = random.choice(real_posts)
        self.post_text = post

        fake_prob = random.randint(50, 100)
        self.output_label = f"{fake_prob}% False"
        if fake_prob > 75:
            self.output_count += 1

    def clear_post_text(self):
        self.post_text = ""
        self.submit_phrase = None 

    @rx.var
    def submit_sequence(self):
        if self.submit_phrase is None:
            return []
        return [
            self.submit_phrase, 1000,
            self.submit_phrase, 1000,
            self.submit_phrase, 1000,
            "Ready"
        ]

    def set_random_phrase(self):
        self.submit_phrase = random.choice(self.submit_phrases)
        self.animation_key += 1

def index() -> rx.Component:
    return rx.container(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("404 Not Found", size="9"),
            rx.text(
                "Final Project - AI BootCamp - May 2025",
                size="5",
            ),
            rx.link(
                rx.button("Get Started!"),
                href="/main",
                is_external=False,
            ),
            spacing="5",
            justify="center",
            min_height="85vh",
        ),
        rx.logo(),
    )

def main_page() -> rx.Component:
    return rx.container(
        rx.heading(
            "Click Bait Buster!",
            font_size="3em",
            margin_bottom="20px",
        ),
        rx.text(
            "Paste a social media post to check its authenticity.",
            margin_bottom="20px",
        ),
        rx.vstack(
            # Text area and Submit button side by side
            rx.hstack(
                rx.text_area(
                    value=State.post_text,
                    placeholder="Enter news link or article text here...",
                    width="500px",
                    height="50px",
                    border="2px solid",
                    border_color="teal",
                    resize="none",
                    on_change=State.set_post_text,
                ),
                rx.button(
                    "Submit",
                    on_click=State.set_random_phrase,
                    color_scheme="green",
                    height="65px",
                    min_width="100px",
                    font_size="1.5em",
                    _hover={"opacity": 0.8},
                ),
                spacing="3",
            ),
            # Generate, Clear, and animated phrases on the same row
            rx.hstack(
                rx.button(
                    "Generate Social Media Post",
                    on_click=State.generate_post,
                    color_scheme="teal",
                ),
                rx.button(
                    "Clear",
                    on_click=State.clear_post_text,
                    color_scheme="red",
                ),
                rx.box(
                    rx.cond(
                        State.submit_sequence != [],
                        type_animation(
                            sequence=State.submit_sequence,
                            speed=50,
                            cursor=True,
                            wrapper="span",
                            style={"font-size": "1em"},
                            key=State.animation_key,
                            repeat=0,
                        ),
                        rx.text(""),
                    ),
                    width="300px",
                    height="40px",
                    padding="10px",
                    display="flex",
                    align_items="right",
                    justify_content="right",
                    text_align="right",
                    margin_left="30px",
                ),
    spacing="3",
),
            # Output and count side by side below
            rx.hstack(
                rx.box(
                    rx.text(
                        State.output_label,
                        font_size="3.5em",
                        font_weight="bold",
                    ),
                    width="300px",
                    height="150px",
                    border="2px solid",
                    border_color="teal",
                    padding="10px",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    text_align="center",
                ),
                rx.box(
                    rx.text(
                        State.output_count_label,
                        font_size="2em",
                        font_weight="bold",
                    ),
                    width="300px",
                    height="150px",
                    border="2px solid",
                    border_color="teal",
                    padding="10px",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    text_align="center",
                ),
                spacing="3",
            ),
            rx.box(
                rx.link(
                    rx.button("Back to Home", color_scheme="teal"),
                    href="/",
                    is_external=False,
                ),
            ),
            spacing="5",
        ),
        spacing="3",
    )

app = rx.App()
app.add_page(index)
app.add_page(main_page, route="/main")

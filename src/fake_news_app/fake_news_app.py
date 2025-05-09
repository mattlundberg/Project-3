import reflex as rx
import random
from rxconfig import config
from reflex_type_animation import type_animation

class State(rx.State):
    output_count: int = 0
    post_text: str = ""
    output_label: str = ""
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
    verdicts = [
        "Pants on Fire",
        "False",
        "Mostly False",
        "Half True",
        "Mostly True"
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

        verdict = random.choice(self.verdicts)
        self.output_label = verdict
        if verdict in ["Pants on Fire", "False", "Mostly False"]:
            self.output_count += 1

    def clear_post_text(self):
        self.post_text = ""
        self.submit_phrase = None 
        self.animation_key = 0
        self.output_label = ""

    @rx.var
    def submit_sequence(self) -> list:
        if self.submit_phrase is None:
            return []
        return [
            self.submit_phrase, 1500,
            "Results Ready..."
        ]

    def set_random_phrase(self):
        self.submit_phrase = random.choice(self.submit_phrases)
        self.animation_key += 1

def index() -> rx.Component:
    return rx.box(
        rx.color_mode.button(position="top-right"),
        rx.center(
            rx.vstack(
                rx.heading(
                    "404 Not Found",
                    size={"base": "7", "md": "9"},
                    style={"font_family": "Lato"},
                    text_align="center",
                    font_weight="bold",
                    color="black",
                ),
                rx.image(
                    src="/Fake-news.jpg",
                    alt="Page not found image",
                    width="900px",
                    height="auto",
                    border_radius="xl"
                ),
                rx.text(
                    size="4",
                    text_align="center",
                    color="gray",
                    style={"font_family": "Lato"},
                ),
                rx.text(
                    "Final Project - AI BootCamp - May 2025",
                    size={"base": "5", "md": "6"},
                    style={"font_family": "Lato"},
                    text_align="center",
                    font_weight="bold",
                    color="black",
                ),
                rx.link(
                    rx.button(
                        "Peel Back the Truth!",
                        style={
                            "font_size": "20px",
                            "font_family": "Lato",
                            "padding": "16px 32px",
                            "background_color": "#D9863B",
                            "color": "white",
                            "border_radius": "8px",
                            "font_weight": "bold",
                        },
                        _hover={
                            "transform": "scale(1.05)",
                            "background_color": "#BA5A31",
                        },
                        transition="all 0.3s ease-in-out",
                    ),
                    href="/main",
                    is_external=False,
                ),
                rx.logo(),
                spacing="4",  
                align="center",
            ),
            min_height="100vh",
        ),
        width="100%",
        min_height="100vh",
        background="linear-gradient(to right, #EAE4D9, #6E5841)",
    )


def main_page() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading(
                "Click Bait Buster!",
                font_size="5em",
                margin_bottom="20px",
                color="black",
            ),
            rx.text(
                "Paste a social media post to check its authenticity.",
                style={"font-size": "1.5em"},
                margin_bottom="20px",
                color="#00695C",
            ),
            rx.vstack(
                rx.hstack(
                    rx.text_area(
                        value=State.post_text,
                        placeholder="Enter social media post here...",
                        width="500px",
                        height="50px",
                        border="3px solid",
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
                                cursor=False,
                                wrapper="span",
                                style={"font-size": "1em"},
                                color="teal",
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
                rx.hstack(
                    rx.box(
                        rx.text(
                            State.output_label,
                            font_size="2.5em",
                            font_weight="bold",
                            color="black",
                        ),
                        width="300px",
                        height="150px",
                        border="3px solid",
                        border_color="teal",
                        background_color="lightgray",
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
                        border="3px solid",
                        border_color="teal",
                        background_color="lightgray",
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
            align_items="center",
        ),
        width="100vw",
        height="100vh",
        background="linear-gradient(to right, #E0F7FA, #80DEEA)",
        justify_content="center",
        align_items="center",
        display="flex",
    )


app = rx.App()
app.add_page(index)
app.add_page(main_page, route="/main")

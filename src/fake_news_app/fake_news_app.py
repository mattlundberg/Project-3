import reflex as rx
import random
from rxconfig import config
from reflex_type_animation import type_animation
from models.modelhelper import ModelHelper
import numpy as np

model_helper = ModelHelper()
loaded_model = model_helper.load_model('text_classification_model_final')

class State(rx.State):
    output_count: int = 0
    post_text: str = ""
    output_label: str = ""
    submit_phrase: str | None = None
    animation_key: int = 0
    image_src: str = "/neutral.png"  # default image
    is_animating: bool = False


    submit_phrases = [
        "Fed Fake News Hamsters...",
        "Combed Through Dumpster X...",
        "Greased the Gears of Progress...",
        "Polished the Data Diamonds...",
        "Woke Up the Server Gremlins...",
        "Tickled the Code Monkeys...",
        "Charged the Quantum Batteries...",
        "Brewed the Algorithm Soup...",
        "Herded Digital Cats...",
        "Spun the Data Wheel...",
        "Sharpened the Logic Knives...",
        "Tuned the Neural Networks...",
        "Stirred the Binary Cauldron...",
        "Trained the Robot Ninjas...",
        "Dusted Off the Server Cobwebs...",
        "Juggled Bits and Bytes...",
        "Chased Down Bug Gremlins...",
        "Wired the Synapse Circuits...",
        "Squeezed the Data Lemons...",
        "Summoned the Code Wizards...",
    ]

    fake_posts_full = [
        "Aliens land in Central Park, demand pizza.",
        "Disney World to lower drinking age to 18 next summer.",
        "Scientists confirm unicorn fossils discovered in Siberia.",
    ]

    real_posts_full = [
        "NASA successfully launches new Mars rover.",
        "WHO recommends regular hand washing to prevent illness.",
        "City council announces plans to expand bike lanes for safer commuting.",
        "FDA approves new gene therapy for treating cystic fibrosis.",
    ]
    fake_posts_pool: list[str] = []
    real_posts_pool: list[str] = []
    
    def get_next_post(self, pool_name: str, full_list_name: str) -> str:
        pool = getattr(self, pool_name)
        full_list = getattr(self, full_list_name)
        if not pool:
            pool = full_list.copy()
            random.shuffle(pool)
        post = pool.pop()
        setattr(self,pool_name,pool)
        return post
    
    def generate_post(self):
        if random.choice([True, False]):
            post = self.get_next_post('fake_posts_pool', 'fake_posts_full')
        else:
            post = self.get_next_post('real_posts_pool', 'real_posts_full')
        self.post_text = post


    def clear_post_text(self):
        self.post_text = ""
        self.submit_phrase = None 
        self.animation_key = 0
        self.output_label = ""
        self.image_src = "/neutral.png"

    @rx.var
    def submit_sequence(self) -> list:
        if self.submit_phrase is None:
            return []
        return [
            self.submit_phrase, 1000,
        ]

    def set_random_phrase(self):
        self.submit_phrase = random.choice(self.submit_phrases)
        self.animation_key += 1
    
    def predict_post(self):
        text = self.post_text
        result = loaded_model.predict(model_helper.preprocess_text(text))
        # Ensure result is a flat array or list
        if hasattr(result, "flatten"):
            result = result.flatten()
        # Get index and value of the highest prediction
        max_index = int(np.argmax(result))
        max_value = float(np.max(result))
        if max_index == 0:
            self.output_label = f"False 🧅"
        elif max_index == 1:
            self.output_label = f"Half True"
        elif max_index == 2:
            self.output_label = f"True 📰"
        else:
            self.output_label = f"Result: {max_index}\nScore: {max_value:.2f}"
        
        if max_index == 2:
            self.image_src = "/positive.png"
        elif max_index == 0:
            self.image_src = "/negative.png"
        else:
            self.image_src = "/neutral.png"

    def submit_and_predict(self):
        self.set_random_phrase()
        self.predict_post()

    def start_animation(self):
        self.set_random_phrase()
        self.is_animating = True
        self.output_label = ""
        self.image_src = "/neutral.png"
        self.animation_key += 1

    def finish_prediction(self):
        self.predict_post()
        self.is_animating = False

    @rx.var
    def output_count_label(self) -> str:
        return f"Count >75%: {self.output_count}"

    def set_post_text(self, value):
        self.post_text = value

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
                "The Onionator!",
                font_size="5em",
                style = {"font_family":"Lato"},
                margin_bottom="20px",
                margin_top = "20px",
                color="black",
            ),
            rx.text(
                "Can you handle the truth?",
                style={
                    "font-size": "1.5em",
                    "font-family":"Lato",
                    },
                margin_bottom="20px",
                color="#00695C",
            ),
            rx.vstack(
                rx.hstack(
                    rx.text_area(
                        value=State.post_text,
                        placeholder="Paste your social media post here...",
                        style = {"font_family":"Lato"},
                        width="500px",
                        height="50px",
                        border="3px solid",
                        border_color="bronze",
                        font = "Lato",
                        resize="none",
                        on_change=State.set_post_text,
                    ),
                    rx.button(
                        "Submit",
                        on_click=State.submit_and_predict,
                        style={
                            "font_size": "20px",
                            "font_family": "Lato",
                            "color": "white",
                            "border_radius": "8px",
                            "font_weight": "bold",
                        },
                        color_scheme="green",
                        height="65px",
                        min_width="100px",
                        font_size="1.5em",
                        font_family="Lato",
                        _hover={"opacity": 0.8},
                    ),
                    spacing="3",
                ),
                rx.hstack(
                    rx.button(
                        "Generate Social Media Post",
                        on_click=State.generate_post,
                        color_scheme="bronze",
                        style={
                            "font_family": "Lato",
                            "background_color": "#D9863B",
                            "color": "white",
                            "border_radius": "8px",
                        },
                    ),
                    rx.button(
                        "Clear",
                        on_click=State.clear_post_text,
                        color_scheme="red",
                        style={
                            "font_family": "Lato",
                            "color": "white",
                            "border_radius": "8px",
                        },
                    ),
                    rx.box(
                        rx.cond(
                            State.submit_sequence != [],
                            type_animation(
                                sequence=State.submit_sequence,
                                speed=50,
                                cursor=False,
                                wrapper="span",
                                style={
                                    "font-size": "1em",
                                    "font-weight": "bold",
                                    "font-family":"Lato"},
                                color="black",
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
                rx.box(
                    rx.text(
                        State.output_label,
                        font_size="2.5em",
                        font_weight="bold",
                        color="black",
                    ),
                    width="600px",
                    height="150px",
                    border="3px solid",
                    border_color="bronze",
                    background_color="lightgray",
                    padding="10px",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    text_align="center",
                    margin_bottom="24px",
                ),
                rx.center(
                    rx.image(
                        src=State.image_src,
                        alt="Result image",
                        width="400px",
                        height="auto",
                        border_radius="xl",
                        style={"background_color": "transparent"},
                    ),
                ),
                rx.center(
                    rx.link(
                        rx.button(
                            "Back to Home",
                            color_scheme="bronze",
                            style={
                                "font_family": "Lato",
                                "background_color": "#D9863B",
                                "color": "white",
                                "border_radius": "8px",
                            },
                        ),
                        href="/",
                        is_external=False,
                    ),
                ),
                spacing="4",
                align_items="center",
            ),
            width="100vw",
            height="100vh",
            background="linear-gradient(to right, #EAE4D9, #6E5841)",
            justify_content="center",
            align_items="center",
            display="flex",
        )
    )


app = rx.App()
app.add_page(index)
app.add_page(main_page, route="/main")

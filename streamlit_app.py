import streamlit as st

from apps import plot_align, plot_multicolor, plot_normal
from utils.multiapp import MultiApp


def main():
    app = MultiApp()

    # Add all your application here
    app.add_app("plot_normal", plot_normal.app)
    app.add_app("plot_align", plot_align.app)
    # app.add_app("plot_multicolor", plot_multicolor.app)

    # The main app
    app.run()


if __name__ == "__main__":
    main()

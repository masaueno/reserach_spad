
import streamlit as st

from utils.multiapp import MultiApp
from apps import plot_normal, plot_align


def main():
    app = MultiApp()

    # Add all your application here
    app.add_app("plot_normal", plot_normal.app)
    app.add_app("plot_align", plot_align.app)


    # The main app
    app.run()

if __name__ == "__main__":
    main()

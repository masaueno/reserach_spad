
import streamlit as st

from utils.multiapp import MultiApp
from apps import plot_spad_all


def main():
    app = MultiApp()

    # Add all your application here
    app.add_app("plot_all", plot_spad_all.app)

    # The main app
    app.run()

if __name__ == "__main__":
    main()

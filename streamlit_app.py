
import streamlit as st

from utils.multiapp import MultiApp
from apps import plot_spad


def main():
    app = MultiApp()

    # Add all your application here
    app.add_app("plot", plot_spad.app)

    # The main app
    app.run()

if __name__ == "__main__":
    main()

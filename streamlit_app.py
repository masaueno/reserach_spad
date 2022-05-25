
import streamlit as st

from utils.multiapp import MultiApp


def main():
    app = MultiApp()

    # Add all your application here
    # app.add_app("Home", home.app)

    # The main app
    app.run()

if __name__ == "__main__":
    main()

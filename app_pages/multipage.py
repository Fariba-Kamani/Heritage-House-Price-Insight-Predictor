import streamlit as st


class MultiPage:
    """
    Class for managing multiple Streamlit pages
    using an object-oriented approach.

    Attributes:
        app_name (str): The title of the Streamlit application.
        pages (list): A list to store page definitions,
                      where each page is a dictionary
                      with keys 'title' (str) and 'function' (callable).
    """

    def __init__(self, app_name) -> None:
        """
        Initializes the MultiPage app with a given name.

        Args:
            app_name (str): The name of the application,
                            displayed at the top of the UI.
        """
        self.pages = []
        self.app_name = app_name

    def add_page(self, title, func) -> None:
        """
        Adds a new page to the application.

        Args:
            title (str): The title to display in the sidebar for this page.
            func (function): The function to call when this page is selected.
        """
        self.pages.append({
            "title": title,
            "function": func
        })

    def run(self):
        """
        Renders the application and allows user
        to select and run a page from the sidebar.
        """
        # Display app title at the top of the page
        st.title(self.app_name)

        # Sidebar menu for selecting pages
        page = st.sidebar.radio(
            label='Menu',
            options=self.pages,
            format_func=lambda page: page['title']
        )

        # Run the selected page's function
        page['function']()

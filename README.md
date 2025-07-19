Starfleet Advisor (TNG Edition) 🚀

A multi-agent conversational AI application that provides Starfleet-themed advice from the crew of the USS Enterprise-D. Users can ask questions on various topics, and the system intelligently routes their query to the most appropriate TNG officer (Captain Picard, Commander Data, Counselor Troi, Commander Riker, or Lt. Commander Geordi La Forge) for a personalized response. The app also includes conversational memory and a custom farewell system.

Features ✨

Multi-Agent Architecture: Leverages LangGraph to orchestrate specialized AI agents.

Linguistic Analyst: Processes user input into formal "Computer Log" entries.

Tactical Officer (Classifier): Dynamically selects the best TNG character to respond based on the query's nature (e.g., ethical, technical, emotional).

Starfleet Advisors: Five distinct TNG personas (Picard, Data, Troi, Riker, Geordi) provide tailored advice.

Conversational Memory: Characters remember previous turns in the conversation for a more coherent dialogue.

Thematic Farewell: Detects "goodbye" intent and responds with a Star Trek-themed closing message.

Streamlit UI: Interactive and user-friendly web interface.

Powered by Google Gemini 1.5 Flash: Utilizes a powerful, fast, and cost-effective LLM.

How It Works ⚙️

The application uses a LangGraph state machine to manage the flow of information and agent interactions:

User Query: The user submits a question via the Streamlit interface.

Linguistic Analyst: The query is first converted into a formal Computer Log entry by a dedicated LLM agent, displayed to the user for thematic effect.

Farewell Detection: A separate LLM agent analyzes the original user query to determine if it's a conversational farewell.

If Yes: The system directly generates a generic Star Trek farewell message and ends the conversation flow.

If No: The flow continues to character selection.

Character Selection: An LLM-powered "Tactical Officer" agent analyzes the Computer Log to identify the core intent (e.g., engineering, ethical, psychological) and intelligently routes the query to the most suitable TNG character.

Character Response: The chosen TNG character (e.g., Picard, Data) receives the original user query (along with the full chat history for context) and generates a personalized response in their distinct persona, directly addressing the user.

Display: The character's response is displayed in the Streamlit chat interface.

Setup and Installation 🛠️

To run this application locally, follow these steps:

Clone the repository:

git clone https://github.com/your-username/starfleet-advisor-tng.git
cd starfleet-advisor-tng

Create a Python virtual environment (recommended):

python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
(You'll need to create a requirements.txt file next.)

Set up your Google API Key:

Obtain a Google API Key from the Google AI Studio.

Create a file named .env in the root of your project directory.

Add your API key to this file:

GOOGLE_API_KEY="YOUR_API_KEY_HERE"

(Optional) For LangSmith Tracing: If you want to debug and visualize the agent's thought process (highly recommended for development):
Create an account on LangSmith.
Get your LangSmith API Key from your settings.

Add these to your .env file as well:

LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY"
LANGCHAIN_PROJECT="starfleet-advisor-tng"

Running the Application 🚀

Once setup is complete, run the Streamlit application from your terminal:
streamlit run app.py
This will open the app in your web browser.

Contributing 🤝

Contributions are welcome! If you have suggestions for new features, bug fixes, or improvements to existing personas, please open an issue or submit a pull request.

License 📄

This project is open-source and available under the MIT License. (You'll need to create a LICENSE file if you choose MIT or another license.)

Acknowledgements 🙏

Built with Google Gemini 2.0 Flash.

Powered by LangChain and LangGraph.

User interface developed with Streamlit.

Inspired by Star Trek: The Next Generation.


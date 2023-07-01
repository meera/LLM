# DIY Planning with AI

This repository contains a Python script that helps users plan DIY projects by providing a list of required tools and supplies along with their costs and links to purchase them from Home Depot. It uses the language understanding capabilities of OpenAI's GPT2 model and SERPApi for fetching product information from Home Depot.

## Installation

Use pip to install the required packages:

```bash
pip install serpapi streamlit langchain
```

## Usage

First, set your SERPAPI API key:

```python
import os
os.environ['SERPAPI_API_KEY'] = 'YOUR_SERPAPI_API_KEY'
```

You can start the application by running this command in your terminal:

```bash
streamlit run shopping.py
```

This will open a web interface in your browser where you can input your project's details.

## How it Works

The application asks for the user's DIY project details. Then it uses OpenAI's language model to generate a list of required tools and supplies. It fetches information about these items from Home Depot using SERPApi and displays the item details (including price and link to purchase) to the user. Finally, it calculates and displays the total cost of all the items.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
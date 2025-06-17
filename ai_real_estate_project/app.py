# --- Python-Dotenv for .env file loading
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import os
import json
import numpy as np
import altair as alt 

# LangChain and Gemini specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Real Estate AI Insights 🏡✨")

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* General Body & Container Styling */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        color: #2c3e50; /* Darker text for readability */
        background-color: #f7f7f7; /* Light grey background */
    }
    .stApp {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px; /* Limit max width for better readability on large screens */
        margin: auto; /* Center the content */
    }

    /* Primary Accent Color */
    :root {
        --primary-color: #FF69B4; /* Pink/Salmon */
        --primary-color-dark: #FF45A6; /* Darker Pink for hover */
        --secondary-text-color: #555;
        --card-background: #ffffff;
        --border-color: #ddd;
        --shadow-light: rgba(0, 0, 0, 0.08);
        --shadow-medium: rgba(0, 0, 0, 0.12);
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
        font-weight: 600; /* Semi-bold headings */
    }

    /* Main Title */
    [data-testid="stHeader"] {
        background-color: transparent; /* Make header transparent */
        color: #2c3e50;
    }
    .st-emotion-cache-h6nsqw { /* Specific Streamlit title class */
        font-size: 2.5em;
        font-weight: 700;
        color: var(--primary-color);
    }

    /* Subheadings */
    .st-emotion-cache-10q2cbe { /* Streamlit's h2/h3 like elements */
        color: #34495e; /* Slightly lighter dark blue */
        font-size: 1.8em;
        font-weight: 600;
    }

    /* Buttons */
    div.stButton > button {
        background-color: var(--primary-color);
        color: white;
        padding: 12px 25px;
        border-radius: 10px; /* Rounded corners */
        border: none;
        font-size: 1.1em;
        font-weight: bold;
        box-shadow: 0 4px 10px var(--shadow-light);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: var(--primary-color-dark);
        box-shadow: 0 6px 15px var(--shadow-medium);
        transform: translateY(-2px);
    }

    /* Text areas and inputs */
    .stTextInput>div>div>textarea, .stTextInput>div>div>input, .stSelectbox>div>div>div.st-cs {
        border-radius: 10px;
        border: 1px solid var(--border-color);
        padding: 10px 15px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTextInput>div>div>textarea:focus, .stTextInput>div>div>input:focus, .stSelectbox>div>div>div.st-cs:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 0.1rem rgba(255, 105, 180, 0.25);
    }

    /* Info/Error/Warning Boxes */
    .stAlert {
        border-radius: 10px;
        padding: 15px;
    }
    .stAlert.info {
        background-color: #e0f2f7; /* Light blue */
        color: #01579b; /* Darker blue */
        border-left: 5px solid #0288d1;
    }
    .stAlert.error {
        background-color: #ffebee; /* Light red */
        color: #c62828; /* Darker red */
        border-left: 5px solid #d32f2f;
    }
    .stAlert.warning {
        background-color: #fffde7; /* Light yellow */
        color: #fbc02d; /* Darker yellow */
        border-left: 5px solid #f9a825;
    }

    /* Feature Tags (Chips) */
    .feature-tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px; /* Space between tags */
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .feature-tag {
        background-color: #e6f7ff; /* Light blue background for tags */
        color: #1890ff; /* Blue text for tags */
        border-radius: 20px; /* More rounded */
        padding: 6px 12px;
        font-size: 0.9em;
        font-weight: 500;
        border: 1px solid #91d5ff; /* Light blue border */
    }
    /* Specific styling for 'Fit Score' container */
    .fit-score-container {
        background-color: var(--card-background);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px var(--shadow-light);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        height: 100%; /* Ensure equal height in columns */
    }
    .fit-score-number {
        font-size: 3em;
        font-weight: bold;
        color: var(--primary-color); /* Pink color for the score */
        margin-bottom: 5px;
    }
    .fit-score-label {
        font-size: 1.1em;
        color: var(--secondary-text-color);
        font-weight: 500;
    }
    /* Style for the general info cards (Buyer Profile, Rationale) */
    .info-card {
        background-color: var(--card-background);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px var(--shadow-light);
        height: 100%; /* Ensure equal height in columns */
    }
    .info-card h4 {
        color: var(--primary-color-dark); /* Darker pink for card titles */
        margin-top: 0;
        font-size: 1.3em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .info-card p {
        color: var(--secondary-text-color);
        font-size: 1em;
        line-height: 1.5;
    }

    /* Expander styling for RAG context */
    [data-testid="stExpander"] {
        background-color: var(--card-background);
        border-radius: 10px;
        padding: 10px 20px;
        box-shadow: 0 2px 10px var(--shadow-light);
        margin-top: 20px;
    }
    [data-testid="stExpander"] div[role="button"] { /* Expander header button */
        font-weight: bold;
        color: #34495e;
    }

    /* Adjust margins for subheaders for better spacing */
    .st-emotion-cache-10q2cbe { /* Target common subheader class */
        margin-top: 3rem; /* More space above sections */
        margin-bottom: 1.5rem;
    }

</style>
""", unsafe_allow_html=True)


# --- API Key Configuration ---
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    # Display an error and stop the app if the API key is not found
    st.error("🚨 **API Key Missing!** Please set `GEMINI_API_KEY` in your `.env` file for local development or in Streamlit Secrets for deployment.")
    st.stop() # Stop the app if API key is not found

# Initialize LangChain's Gemini integration
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key, temperature=0.7)
output_parser = StrOutputParser() # To get raw string output from LLM directly from LLM response

# --- Data Loading and Preprocessing ---
@st.cache_data # Cache data loading for performance across reruns
def load_and_process_ny_house_data():
    """
    Loads the NY-House-Dataset.csv, performs essential preprocessing,
    and computes neighborhood hotness scores.
    """
    try:
        df_houses = pd.read_csv('NY-House-Dataset.csv')

        # Basic cleaning and type conversion for numerical columns
        df_houses['BATH'] = df_houses['BATH'].fillna(df_houses['BATH'].median())
        df_houses['PROPERTYSQFT'] = pd.to_numeric(df_houses['PROPERTYSQFT'], errors='coerce').fillna(0) 
        df_houses['PRICE'] = pd.to_numeric(df_houses['PRICE'], errors='coerce').fillna(0)

        df_houses['NEIGHBORHOOD'] = df_houses['LOCALITY'].fillna('') + ' - ' + df_houses['SUBLOCALITY'].fillna('')
        df_houses['NEIGHBORHOOD'] = df_houses['NEIGHBORHOOD'].str.strip(' -')
        df_houses['NEIGHBORHOOD'] = df_houses['NEIGHBORHOOD'].replace('', np.nan)
        df_houses['NEIGHBORHOOD'] = df_houses['NEIGHBORHOOD'].fillna(df_houses['ADMINISTRATIVE_AREA_LEVEL_2'].fillna('Unknown Area'))

        # --- Compute Neighborhood Hotness Score ---
        neighborhood_stats = df_houses.groupby('NEIGHBORHOOD').agg(
            median_price=('PRICE', 'median'),
            avg_beds=('BEDS', 'mean'),
            avg_baths=('BATH', 'mean'),
            avg_sqft=('PROPERTYSQFT', 'mean'),
            num_listings=('BROKERTITLE', 'count')
        ).reset_index()

        min_listings_threshold = 10
        neighborhood_stats = neighborhood_stats[neighborhood_stats['num_listings'] >= min_listings_threshold]

        if not neighborhood_stats.empty:
            price_range = neighborhood_stats['median_price'].max() - neighborhood_stats['median_price'].min()
            listings_range = neighborhood_stats['num_listings'].max() - neighborhood_stats['num_listings'].min()

            neighborhood_stats['normalized_price'] = 0.0
            if price_range > 0:
                neighborhood_stats['normalized_price'] = (neighborhood_stats['median_price'] - neighborhood_stats['median_price'].min()) / price_range
            
            neighborhood_stats['normalized_listings'] = 0.0
            if listings_range > 0:
                neighborhood_stats['normalized_listings'] = (neighborhood_stats['num_listings'] - neighborhood_stats['num_listings'].min()) / listings_range
            
            neighborhood_stats['hotness_score'] = (neighborhood_stats['normalized_price'] * 0.6 + \
                                                  neighborhood_stats['normalized_listings'] * 0.4) * 100

            top_hot_neighborhoods = neighborhood_stats.sort_values(by='hotness_score', ascending=False).head(10)
        else:
            top_hot_neighborhoods = pd.DataFrame()

        return df_houses, top_hot_neighborhoods

    except FileNotFoundError:
        st.error("🚨 **Error:** `NY-House-Dataset.csv` not found. Please ensure it's in the same directory as `app.py`.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 **Data Processing Error:** {e}. Please check the CSV file format and its contents.")
        st.stop()

# Load and process data when the Streamlit app starts
df_houses, df_hot_neighborhoods = load_and_process_ny_house_data()


# --- Conceptual RAG Setup ---

@st.cache_data # Cache document preparation for performance
def prepare_rag_documents(df: pd.DataFrame) -> list[str]:
    """
    Prepares a list of concise string summaries of properties from the DataFrame
    for conceptual RAG retrieval.
    """
    documents = []
    sample_df = df.sample(min(2000, len(df)), random_state=42) # Sample up to 2000 documents
    
    for index, row in sample_df.iterrows():
        price_str = f"${row['PRICE']:,}" if pd.notna(row['PRICE']) and row['PRICE'] > 0 else "Price N/A"
        beds_str = f"{int(row['BEDS'])} Beds" if pd.notna(row['BEDS']) else "Beds N/A"
        baths_str = f"{row['BATH']:.1f} Baths" if pd.notna(row['BATH']) else "Baths N/A"
        sqft_str = f"{row['PROPERTYSQFT']:.0f} SqFt" if pd.notna(row['PROPERTYSQFT']) and row['PROPERTYSQFT'] > 0 else "SqFt N/A"
        
        doc = (
            f"Property Listing Details:\n"
            f"- Type: {row['TYPE']}\n"
            f"- Price: {price_str}\n"
            f"- Bedrooms: {beds_str}\n"
            f"- Bathrooms: {baths_str}\n"
            f"- Square Footage: {sqft_str}\n"
            f"- Address: {row['FORMATTED_ADDRESS']}\n"
            f"- Neighborhood: {row['NEIGHBORHOOD']}\n"
            f"- Broker: {row['BROKERTITLE']}\n"
            f"This property is in {row['LOCALITY']}, {row['ADMINISTRATIVE_AREA_LEVEL_2']}, New York."
        )
        documents.append(doc)
    return documents

rag_documents = prepare_rag_documents(df_houses)

def retrieve_context_conceptual(query_desc: str, df_data: pd.DataFrame = df_houses, top_k: int = 3) -> str:
    """
    Simulates retrieval for RAG based on keyword matching and basic price filtering.
    """
    relevant_properties_info = []
    query_words = set(query_desc.lower().split())

    meaningful_query_words = {word for word in query_words if len(word) > 2 and word not in ['a', 'an', 'the', 'is', 'are', 'for', 'in', 'with', 'and', 'or', 'of', 'to', 'from', 'house', 'home', 'apartment', 'condo', 'townhouse']}

    filtered_df = df_data.copy()

    if meaningful_query_words:
        search_cols = ['ADDRESS', 'TYPE', 'NEIGHBORHOOD', 'LOCALITY', 'SUBLOCALITY', 'ADMINISTRATIVE_AREA_LEVEL_2']
        filtered_df['SEARCH_TEXT'] = filtered_df[search_cols].fillna('').agg(' '.join, axis=1).str.lower()
        
        keyword_pattern = '|'.join(meaningful_query_words)
        if keyword_pattern:
            filtered_df = filtered_df[filtered_df['SEARCH_TEXT'].str.contains(keyword_pattern, na=False)]
        filtered_df = filtered_df.drop(columns=['SEARCH_TEXT'])


    query_lower = query_desc.lower()
    
    median_price = df_data['PRICE'].median()
    q25_price = df_data['PRICE'].quantile(0.25)
    q75_price = df_data['PRICE'].quantile(0.75)

    if "cheap" in query_lower or "affordable" in query_lower or "low price" in query_lower:
        filtered_df = filtered_df[filtered_df['PRICE'] <= q25_price]
    elif "expensive" in query_lower or "luxury" in query_lower or "high-end" in query_lower or "high price" in query_lower:
        filtered_df = filtered_df[filtered_df['PRICE'] >= q75_price]

    if not filtered_df.empty:
        filtered_df['PRICE'] = pd.to_numeric(filtered_df['PRICE'], errors='coerce')
        filtered_df = filtered_df.dropna(subset=['PRICE'])

        if "cheap" in query_lower or "affordable" in query_lower or "low price" in query_lower:
            sampled_matches = filtered_df.sort_values(by='PRICE', ascending=True).drop_duplicates(subset=['FORMATTED_ADDRESS']).head(top_k)
        elif "expensive" in query_lower or "luxury" in query_lower or "high-end" in query_lower or "high price" in query_lower:
             sampled_matches = filtered_df.sort_values(by='PRICE', ascending=False).drop_duplicates(subset=['FORMATTED_ADDRESS']).head(top_k)
        else:
            sampled_matches = filtered_df.sort_values(by='PROPERTYSQFT', ascending=False).drop_duplicates(subset=['FORMATTED_ADDRESS']).head(top_k)

        for _, row in sampled_matches.iterrows():
            price_str = f"${row['PRICE']:,}" if pd.notna(row['PRICE']) else "Price N/A"
            beds_str = f"{int(row['BEDS'])}" if pd.notna(row['BEDS']) else "N/A"
            baths_str = f"{row['BATH']:.1f}" if pd.notna(row['BATH']) else "N/A"
            sqft_str = f"{row['PROPERTYSQFT']:.0f}" if pd.notna(row['PROPERTYSQFT']) else "N/A"

            relevant_properties_info.append(
                f"- Address: {row['FORMATTED_ADDRESS']}; Type: {row['TYPE']}; Price: {price_str}; "
                f"Beds: {beds_str}; Baths: {baths_str}; SqFt: {sqft_str}."
            )
    
    if relevant_properties_info:
        return "Retrieved relevant property information from our database for context:\n" + "\n".join(relevant_properties_info)
    else:
        if "cheap" in query_lower or "affordable" in query_lower or "low price" in query_lower:
            return "No very cheap properties found matching your criteria in our database for context. Try a more general description or different keywords."
        elif "expensive" in query_lower or "luxury" in query_lower or "high-end" in query_lower or "high price" in query_lower:
            return "No very expensive properties found matching your criteria in our database for context. Try a more general description or different keywords."
        return "No highly relevant property context retrieved from database based on your keywords."

# --- LangChain Prompt Template for Property Analyzer ---
property_analyzer_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a highly analytical real estate AI assistant for NextBurb. Your goal is to provide concise, structured insights about a property based on its description and provided context from our internal database.
    Return your response STRICTLY as a JSON string with the following keys:
    'features': Markdown bullet points of key property features (e.g., '3 Beds, 2 Baths', 'Renovated kitchen').
    'buyer_profile': 1-2 sentence ideal buyer assessment.
    'fit_score': Integer score out of 100 for general appeal.
    'rationale': 1-2 sentence explanation for the fit_score, referencing provided context if applicable and the property's features.

    Retrieved Property Context from Internal Database (if available and relevant to the query):
    {context}
    """),
    ("human", "Analyze the following property description: {description}")
])

# --- Functions for AI Processing ---
def get_property_insights_gemini(description: str):
    context = retrieve_context_conceptual(description, df_data=df_houses)
    st.info(f"**Context Retrieved for RAG (from Dataset):**\n```\n{context}\n```")

    chain = property_analyzer_prompt_template | llm | output_parser

    try:
        response = chain.invoke({"context": context, "description": description})
    except Exception as e:
        st.error(f"🚨 **Error invoking Gemini API:** {e}. Please check your API key and internet connection.")
        return {
            "features": "- API call failed.",
            "buyer_profile": "API call failed.",
            "fit_score": 0,
            "rationale": "Could not connect to AI service or API error. Check terminal for details."
        }

    try:
        if response.strip().startswith("```json"):
            response = response.strip()[7:].strip()
            if response.endswith("```"):
                response = response[:-3].strip()
        
        parsed_json = json.loads(response)
        
        required_keys = ['features', 'buyer_profile', 'fit_score', 'rationale']
        if not all(key in parsed_json for key in required_keys):
            raise ValueError(f"LLM response missing required JSON keys. Missing: {', '.join([k for k in required_keys if k not in parsed_json])}")
            
        if not isinstance(parsed_json.get('fit_score'), int):
            try:
                parsed_json['fit_score'] = int(parsed_json.get('fit_score', 0))
            except ValueError:
                parsed_json['fit_score'] = 0
        
        return parsed_json
    except json.JSONDecodeError as e:
        st.error(f"🚨 **Error parsing AI response JSON.** Raw response from LLM:\n```\n{response}\n```\nError: {e}")
        return {
            "features": "- Error: Could not parse features. Raw response was not valid JSON.",
            "buyer_profile": "Error: Could not parse ideal buyer profile.",
            "fit_score": 0,
            "rationale": "AI response was not in expected JSON format. Please try again or refine input. (See raw response above for debugging)."
        }
    except Exception as e:
        st.error(f"🚨 **An unexpected error occurred during AI response processing:** {e}")
        return {
            "features": "- Error: An unexpected error occurred.",
            "buyer_profile": "Error: An unexpected error occurred.",
            "fit_score": 0,
            "rationale": "An unexpected error occurred during AI response processing. (Check terminal for details)."
        }


# --- Streamlit UI Layout ---

# Page Title & Description
st.title("🏡 Real Estate AI Insights Dashboard")
st.write("Your Smart Guide to NYC Real Estate.")

st.markdown("---") # Visual separator

# --- Quick Listing Analyzer Section ---
st.subheader("🤖 Quick Listing Analyzer")
# Add clearer instructions and examples
st.markdown("""
To get started, **paste a full property listing description** below.
Our AI will analyze it and provide key features, an ideal buyer profile, and a 'Fit Score'.
<br>
**Example:** `Spacious 3-bed, 2-bath condo in Manhattan with river views. Features a modern kitchen, large living area, and access to a gym. Building has a concierge. Close to Central Park and subway line A. Ideal for city living.`
""", unsafe_allow_html=True)


property_description_analyzer = st.text_area(
    "Paste Listing Description here:",
    "", # Start with empty value for user input
    height=150,
    key="analyzer_input", # Unique key for this widget
    # Placeholder text is now redundant with markdown instructions above
)

# Button to trigger analysis
if st.button("Analyze Listing", key="analyze_button"):
    if property_description_analyzer:
        with st.spinner("Analyzing listing with Gemini AI and retrieving context..."):
            insights = get_property_insights_gemini(property_description_analyzer)
            
            # --- Display Listing Analysis Results (Enhanced UI) ---
            st.markdown("#### Listing Analysis Results:")
            
            # Feature Tags
            if 'features' in insights and insights['features']:
                features_list = [f.strip('- ').strip() for f in insights['features'].split('\n') if f.strip()]
                st.markdown('<div class="feature-tag-container">', unsafe_allow_html=True)
                for feature in features_list:
                    st.markdown(f'<span class="feature-tag">{feature}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No specific features extracted by AI.")

            # Buyer Profile, Fit Score, Rationale in columns
            col_buyer, col_fit, col_rationale = st.columns([1.5, 1, 2]) # Adjust column ratios

            with col_buyer:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown("<h4>🧑‍💻 Buyer Profile</h4>", unsafe_allow_html=True)
                st.markdown(f"<p>{insights.get('buyer_profile', 'N/A')}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_fit:
                st.markdown('<div class="fit-score-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="fit-score-number">✅ {insights.get("fit_score", 0)}/100</div>', unsafe_allow_html=True)
                st.markdown('<div class="fit-score-label">Fit Score</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_rationale:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown("<h4>📝 Rationale</h4>", unsafe_allow_html=True)
                st.markdown(f"<p>{insights.get('rationale', 'N/A')}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # RAG Context Expander
            with st.expander("⬆️ Context Retrieved for RAG (click to expand)"):
                st.write("This section shows the data retrieved from the `NY-House-Dataset.csv` that was used by the AI to inform its analysis.")
                # The actual context content is displayed via st.info just above this expander in get_property_insights_gemini

            st.markdown("---") # Visual separator after results
    else:
        st.warning("Please enter a property description to analyze.")

st.markdown("---") # Visual separator before next section

# --- Neighborhood Insights & "Hotness" Analysis (Dataset-Driven) ---
st.subheader("📈 NYC Neighborhood Hotness Insights")
st.markdown("Explore top 'hot' neighborhoods in New York based on aggregated data derived from our provided dataset. Use the filters below to refine your view.")

# Filters for the chart
col_chart_filter1, col_chart_filter2 = st.columns(2)

with col_chart_filter1:
    # Get unique localities for filter, ensuring no empty/NaN and sorting alphabetically
    chart_localities = ['All'] + sorted([loc for loc in df_houses['LOCALITY'].dropna().unique().tolist() if loc and loc != 'Unknown Area'])
    selected_locality = st.selectbox("Filter by Locality/Borough:", chart_localities, key="chart_locality_filter")

with col_chart_filter2:
    chart_price_ranges = ['All', 'Under $500,000', '$500,000 - $1,000,000', '$1,000,000 - $2,000,000', '$2,000,000 - $5,000,000', 'Over $5,000,000']
    selected_price_range = st.selectbox("Filter by Price Range:", chart_price_ranges, key="chart_price_filter")

# Apply filters to hot neighborhoods data
filtered_hot_neighborhoods = df_hot_neighborhoods.copy()

if selected_locality != 'All':
    # Filter hot neighborhoods based on if their name contains the selected locality
    filtered_hot_neighborhoods = filtered_hot_neighborhoods[
        filtered_hot_neighborhoods['NEIGHBORHOOD'].str.contains(selected_locality, case=False, na=False)
    ]

if selected_price_range != 'All':
    # Filter based on median price falling into the selected range
    if selected_price_range == 'Under $500,000':
        filtered_hot_neighborhoods = filtered_hot_neighborhoods[filtered_hot_neighborhoods['median_price'] <= 500000]
    elif selected_price_range == '$500,000 - $1,000,000':
        filtered_hot_neighborhoods = filtered_hot_neighborhoods[(filtered_hot_neighborhoods['median_price'] > 500000) & (filtered_hot_neighborhoods['median_price'] <= 1000000)]
    elif selected_price_range == '$1,000,000 - $2,000,000':
        filtered_hot_neighborhoods = filtered_hot_neighborhoods[(filtered_hot_neighborhoods['median_price'] > 1000000) & (filtered_hot_neighborhoods['median_price'] <= 2000000)]
    elif selected_price_range == '$2,000,000 - $5,000,000':
        filtered_hot_neighborhoods = filtered_hot_neighborhoods[(filtered_hot_neighborhoods['median_price'] > 2000000) & (filtered_hot_neighborhoods['median_price'] <= 5000000)]
    elif selected_price_range == 'Over $5,000,000':
        filtered_hot_neighborhoods = filtered_hot_neighborhoods[filtered_hot_neighborhoods['median_price'] > 5000000]


if not filtered_hot_neighborhoods.empty:
    # Use Altair for a more visually appealing and customizable bar chart
    # Create the chart with specified encoding and interactivity
    chart = alt.Chart(filtered_hot_neighborhoods).mark_bar(color='#FF69B4').encode(
        x=alt.X('hotness_score', title='Hotness Score'),
        y=alt.Y('NEIGHBORHOOD', sort='-x', title='Neighborhood'), # Sort neighborhoods by hotness score descending
        tooltip=['NEIGHBORHOOD', 'median_price', 'num_listings', alt.Tooltip('hotness_score', format='.1f')] # Tooltips on hover
    ).properties(
        title='Top Neighborhood Hotness by Score' # Chart title
    ).interactive() # Enable zooming and panning

    st.altair_chart(chart, use_container_width=True) # Display the Altair chart

    st.markdown("""
    <small><i>* 'Hotness Score' is a prototype metric based on median property price and number of listings.</i></small>
    """, unsafe_allow_html=True)
else:
    st.info("No hot neighborhoods found matching the selected filters. Try adjusting your criteria.")

st.markdown("---") # Visual separator

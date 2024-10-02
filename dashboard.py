import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

#Streamlit page configuration
st.set_page_config(layout="centered", page_title="Data Analysis Project with E-Commerce Public Dataset")

def load_data():
    # Load ataset
    return pd.read_csv('merged_data.csv')

data = load_data()

# Convert order_purchase_timestamp to datetime format
data['order_purchase_timestamp'] = pd.to_datetime(data['order_purchase_timestamp'])

# Sidebar for filtering data by date range
st.sidebar.header("Filter by Date")
min_date = pd.to_datetime(data['order_purchase_timestamp']).min().date()
# Allow max date to be today without limiting
max_date = datetime.now().date()

start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=start_date)

# Filter data based on the selected date range
filtered_data = data[(data['order_purchase_timestamp'].dt.date >= start_date) & 
                     (data['order_purchase_timestamp'].dt.date <= end_date)]


# Tittle of the dashboard
st.title("Data Analysis Project with E-Commerce Public Dataset")

# List of contents with anchor links
st.markdown("""
## Contents
- [Top Product Categories by Average Transaction Value](#top-product-categories-by-average-transaction-value)
- [Delivery Time vs Customer Satisfaction](#delivery-time-vs-customer-satisfaction)
- [RFM Segmentation](#rfm-segmentation)
- [RFM Segmentation Groups](#rfm-segmentation-groups)
- [Customer Distribution by Monetary Groups](#customer-distribution-by-monetary-groups)
""")

# --- Top Product Categories by Average Transaction Value ---
st.header("Top Product Categories by Average Transaction Value")

# Top N selection and sort order
col1, col2 = st.columns([1, 1])

with col1:
    top_n = st.number_input("Select Top N Categories", min_value=1, max_value=20, value=5)

with col2:
    sort_order = st.radio("Sort Order", ('Descending', 'Ascending'))

# Group by product category and calculate average transaction value (mean of total_price)
top_product_categories = (filtered_data.groupby('product_category_name_english')
                          .agg({'total_price': 'mean'})
                          .sort_values('total_price', ascending=(sort_order == 'Ascending'))
                          .head(top_n)
                          .reset_index())

# Rename column for display purposes
top_product_categories.rename(columns={'product_category_name_english': 'Product Category', 'total_price': 'Average Transaction Value'}, inplace=True)

# Display the table
st.dataframe(top_product_categories)

# Plot the top product categories
plt.figure(figsize=(8, 4))
sns.barplot(data=top_product_categories, x='Product Category', y='Average Transaction Value', palette='viridis')
plt.title('Top Product Categories by Average Transaction Value')
plt.xticks(rotation=45)
st.pyplot(plt)

# --- Delivery Time vs Customer Satisfaction ---
st.header("Delivery Time vs Customer Satisfaction")

# Create a DataFrame with delivery time and review score
delivery_vs_rating = filtered_data[['order_purchase_timestamp', 'order_delivered_customer_date', 'review_score']].dropna()
delivery_vs_rating['delivery_time'] = (pd.to_datetime(delivery_vs_rating['order_delivered_customer_date']) - 
                                       pd.to_datetime(delivery_vs_rating['order_purchase_timestamp'])).dt.days

# Add range selection for delivery time
delivery_time_range = st.slider("Select Delivery Time Range (Days)", min_value=0, max_value=100, value=(0, 50), step=50)

# Filter data based on delivery time range
delivery_vs_rating_filtered = delivery_vs_rating[(delivery_vs_rating['delivery_time'] >= delivery_time_range[0]) & 
                                                 (delivery_vs_rating['delivery_time'] <= delivery_time_range[1])]

# Scatter plot of delivery time vs review score
plt.figure(figsize=(8, 4))
sns.scatterplot(data=delivery_vs_rating_filtered, x='delivery_time', y='review_score', alpha=0.6)
plt.title('Delivery Time vs Customer Satisfaction')
plt.xlabel('Delivery Time (Days)')
plt.ylabel('Review Score')
st.pyplot(plt)

# --- RFM Calculation and Segmentation ---
st.header("RFM (Recency, Frequency, Monetary) Segmentation")

# Get the reference date (1 day after the most recent purchase in the filtered dataset)
reference_date = filtered_data['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

# Calculate Recency, Frequency, and Monetary metrics
rfm_data = filtered_data.groupby('customer_id').agg({
    'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,  # Recency
    'order_id': 'count',  # Frequency
    'total_price': 'sum'  # Monetary
}).reset_index().rename(columns={
    'order_purchase_timestamp': 'Recency',
    'order_id': 'Frequency',
    'total_price': 'Monetary'
})

# Display RFM data with customizable number of rows
num_rows_rfm = st.number_input("Number of rows to display in RFM Table", min_value=1, max_value=len(rfm_data), value=5)

st.write(f"RFM Table (First {num_rows_rfm} Rows):")
st.dataframe(rfm_data.head(num_rows_rfm))

# RFM Segmentation
st.header("RFM Segmentation Groups")

# Define bins for Recency, Frequency, and Monetary
recency_bins = [0, 30, 60, 90, 180, float('inf')]
recency_labels = ['Very Recent', 'Recent', 'Moderate', 'Old', 'Very Old']

frequency_bins = [0, 1, 5, 10, 20, float('inf')]
frequency_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

monetary_bins = [0, 100, 500, 1000, 5000, float('inf')]
monetary_labels = ['Low', 'Medium', 'High', 'Very High', 'Premium']

# Assign RFM segments based on the bins
rfm_data['Recency_Group'] = pd.cut(rfm_data['Recency'], bins=recency_bins, labels=recency_labels, right=False)
rfm_data['Frequency_Group'] = pd.cut(rfm_data['Frequency'], bins=frequency_bins, labels=frequency_labels, right=False)
rfm_data['Monetary_Group'] = pd.cut(rfm_data['Monetary'], bins=monetary_bins, labels=monetary_labels, right=False)

# Display the segmented RFM Table
st.write(f"Segmented RFM Table (First {num_rows_rfm} Rows):")
st.dataframe(rfm_data[['customer_id', 'Recency', 'Frequency', 'Monetary', 'Recency_Group', 'Frequency_Group', 'Monetary_Group']].head(num_rows_rfm))

# RFM Segmentation Heatmap
st.header("RFM Segments Heatmap (Recency vs Frequency)")

# Create a pivot table for heatmap visualization
rfm_segment_counts = rfm_data.groupby(['Recency_Group', 'Frequency_Group']).size().reset_index(name='Count')
heatmap_data = rfm_segment_counts.pivot_table(index='Recency_Group', columns='Frequency_Group', values='Count', fill_value=0)

# Plot heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(heatmap_data, annot=True, cmap='Blues', fmt='g')
plt.title('Heatmap of RFM Segments')
plt.xlabel('Frequency Group')
plt.ylabel('Recency Group')
st.pyplot(plt)

# Customer Distribution by Monetary Group
st.header("Customer Distribution by Monetary Groups")

# Count the number of customers in each Monetary Group
monetary_counts = rfm_data['Monetary_Group'].value_counts().sort_index()

# Create a DataFrame from the counts for easier plotting
monetary_df = monetary_counts.reset_index()
monetary_df.columns = ['Monetary_Group', 'Customer_Count']

# Create a bar plot for the Monetary Groups
plt.figure(figsize=(8, 4))
sns.barplot(data=monetary_df, x='Monetary_Group', y='Customer_Count', palette='viridis')
plt.title('Customer Distribution by Monetary Groups')
plt.xlabel('Monetary Group')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
st.pyplot(plt)

# Todo
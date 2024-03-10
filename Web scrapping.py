#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import requests


from bs4 import BeautifulSoup   #used for web scraping


# In[54]:


import requests
import urllib3
import ssl


class CustomHttpAdapter (requests.adapters.HTTPAdapter):
    # "Transport adapter" that allows us to use custom ssl_context.

    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections, maxsize=maxsize,
            block=block, ssl_context=self.ssl_context)


def get_legacy_session():
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
    session = requests.session()
    session.mount('https://', CustomHttpAdapter(ctx))
    return session


# In[21]:


get_legacy_session().get("https://tools.niehs.nih.gov/stdb/index.cfm?do=spintrap.search&radical=OH&search=true&spintrap=DMPO&page=1")


# In[22]:


response=get_legacy_session().get("https://tools.niehs.nih.gov/stdb/index.cfm?do=spintrap.search&radical=OH&search=true&spintrap=DMPO&page=1").text


# In[23]:


response


# In[24]:


soup=BeautifulSoup(response,'lxml')  # create object, lxml helps in parsing the html file


# In[25]:


print(soup.prettify())  #to easily understand html structure


# In[26]:


#extract h1 tags
soup.find_all('h1')[0].text.strip()


# In[27]:


for i in soup.find_all('h2'):
    print(i.text.strip())


# In[28]:


table=soup.find_all('table', class_='dataTable ui-responsive ui-table large-only')  # use class 


# In[29]:


# Iterate over elements in the table
for table_element in table:
    # Apply find() method to each element
    tbody = table_element.find('tbody')
    # Process the found tbody element
    
    
    if tbody:
        # Do something with the tbody element
        print(tbody)


# In[57]:


# Initialize lists to store extracted data
all_rows_data = []

# Iterate over each page
for page_num in range(1, 19):
    # Make a request to the page
    url = f"https://tools.niehs.nih.gov/stdb/index.cfm?do=spintrap.search&radical=OH&search=true&spintrap=DMPO&page={page_num}"  # Replace with the actual URL pattern
    response=get_legacy_session().get(url)
    
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the tbody element containing the data
        tbody = soup.find('tbody')
        if tbody:
            # Extract information from each row (tr) within the tbody
            rows = tbody.find_all('tr')
            for row in rows:
                # Initialize a dictionary to store data of each row
                row_data = {}
                
                # Extract information from each cell (td) within the row
                cells = row.find_all('td')
                
                # Extract and store data from each cell
                for index, cell in enumerate(cells):
                    # Assuming index 0 contains the first column data, index 1 contains the second column data, and so on
                    if index == 0:
                        # Split the cell content to extract the spin trap and comments
                        cell_text = cell.get_text(strip=True)
                        #split_text = cell_text.split(' in ')
                        row_data['Spin Trap with comments'] = cell_text
                        #row_data['Comments'] = split_text[1]
                    elif index == 1:
                        # Extract and store aN value
                        aN_text = cell.get_text(strip=True)
                        if 'N/A' in aN_text:
                            row_data['aN'] = None
                        else:
                            row_data['aN'] = float(aN_text.split('=')[1].split('G')[0])
                    elif index == 2:
                        # Extract and store aH value
                        aH_text = cell.get_text(strip=True)
                        if 'N/A' in aH_text:
                            row_data['aH'] = None
                        else:
                            row_data['aH'] = float(aH_text.split('=')[1].split('G')[0])
                    elif index == 3:
                        # Extract and store the aN/aH value
                        aN_aH_text = cell.get_text(strip=True)
                        if 'N/A' in aN_aH_text:
                            row_data['aN/aH'] = None
                        else:
                            row_data['aN/aH'] = float(aN_aH_text)
                    elif index == 4:
                        # Extract and store comments
                        row_data['Comments'] = cell.get_text(strip=True)
                # Add the "Result" column with value "OH"
                row_data['Spin Trap'] = "OH"
                # Append the data of the current row to the list of rows data
                all_rows_data.append(row_data)
    else:
        print(f"Failed to fetch page {page_num}")

# Create a DataFrame from the extracted data
df = pd.DataFrame(all_rows_data)

# Print the DataFrame
print(df)


# In[58]:


# Initialize lists to store extracted data
rows_data = []

# Iterate over each page
for page_num in range(1, 19):
    # Make a request to the page
    url = f"https://tools.niehs.nih.gov/stdb/index.cfm?do=spintrap.search&radical=O2H&search=true&spintrap=DMPO&page={page_num}"  # Replace with the actual URL pattern
    response=get_legacy_session().get(url)
    
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the tbody element containing the data
        tbody = soup.find('tbody')
        if tbody:
            # Extract information from each row (tr) within the tbody
            rows = tbody.find_all('tr')
            for row in rows:
                # Initialize a dictionary to store data of each row
                row_data = {}
                
                # Extract information from each cell (td) within the row
                cells = row.find_all('td')
                
                # Extract and store data from each cell
                for index, cell in enumerate(cells):
                    # Assuming index 0 contains the first column data, index 1 contains the second column data, and so on
                    if index == 0:
                        # Split the cell content to extract the spin trap and comments
                        cell_text = cell.get_text(strip=True)
                        #split_text = cell_text.split(' in ')
                        row_data['Spin Trap with comments'] = cell_text
                        #row_data['Comments'] = split_text[1]
                    elif index == 1:
                        # Extract and store aN value
                        aN_text = cell.get_text(strip=True)
                        if 'N/A' in aN_text:
                            row_data['aN'] = None
                        else:
                            row_data['aN'] = float(aN_text.split('=')[1].split('G')[0])
                    elif index == 2:
                        # Extract and store aH value
                        aH_text = cell.get_text(strip=True)
                        if 'N/A' in aH_text:
                            row_data['aH'] = None
                        else:
                            row_data['aH'] = float(aH_text.split('=')[1].split('G')[0])
                    elif index == 3:
                        # Extract and store the aN/aH value
                        aN_aH_text = cell.get_text(strip=True)
                        if 'N/A' in aN_aH_text:
                            row_data['aN/aH'] = None
                        else:
                            row_data['aN/aH'] = float(aN_aH_text)
                    elif index == 4:
                        # Extract and store comments
                        row_data['Comments'] = cell.get_text(strip=True)
                # Add the "Result" column with value "O2H"
                row_data['Spin Trap'] = "O2H"
                # Append the data of the current row to the list of rows data
                rows_data.append(row_data)
    else:
        print(f"Failed to fetch page {page_num}")

# Create a DataFrame from the extracted data
df1 = pd.DataFrame(rows_data)

# Print the DataFrame
print(df1)


# In[59]:


# Concatenate both DataFrames
merged_df = pd.concat([df, df1], ignore_index=True)


# In[60]:


# Shuffle the rows of the merged DataFrame
merged_df = merged_df.sample(frac=1).reset_index(drop=True)
merged_df


# In[61]:


# Write the merged DataFrame to a CSV file
merged_df.to_csv('merged_data.csv', index=False)


# In[ ]:





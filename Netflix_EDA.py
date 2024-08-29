'''This project involves loading, cleaning, analyzing, and visualizing data from a Netflix dataset. We'll use Python libraries like Pandas, Matplotlib, and Seaborn to work through the project.
   The goal is to explore the dataset, derive insights, and prepare for potential machine learning tasks'''

#importing libraries and loading dataframe
import pandas as pd
import mathplotlib.pyplot as plt
import seaborn as sns

#Loading the Data
d=pd.read_csv("/content/netflix1.csv")
df=pd.DataFrame(d)
#display first few rows of dataset to know about the contents and attributes
print(df.head(10))
#Lets get some info about the data 
df.info()

#Cleaning the Dataset
#1.Checking for missing values and dropping the rows with missing values
df.isnull().sum()#There's no columns with missing values so we don't need to drop any rows

#Here we can see a lot of rows has 'Not Given' in it instesd of any NULL or NA, so to remove them we will use
df.map(lambda x: x.strip().lower() if isinstance(x, str) else x)
df= df[~df.apply(lambda row: row.str.lower().str.contains('not given').any(), axis=1)]


#2.Only check if movie Title and Show ID is duplicated and drop the rows with duplicates
df.duplicated(subset=["show_id"])
df.drop_duplicates(subset=["show_id"],inplace=True)
df.duplicated(subset=["title"])
df.drop_duplicates(subset=["title"],inplace=True)

#3.Convert the data type of date_added column to date time from object type
df['date_added']=pd.to_datatime(df['data_added'])
df.dtype #to check if data type of date_added has changed

#Data Analyzation
#1.counting the number of movies and tv shows
type_counts=df['type'].value_counts()
#since above is a Series object lets convert it into a df
df1=pd.DataFrame({'type':['Movie','TV Show'],'count':[6125,2663]})

#2.splitting the listed_in column into a list of genres by using lamda fucn that separates eachcomma separated string into a list of strings
df['genres'] = df['listed_in'].apply(lambda x: x.split(','))
#concatenate the list of genres of each rows into a single 1-D list using sum func for strings
all_genres = sum(df['genres'], [])
#counting the number of genres
genre_counts = pd.Series(all_genres).value_counts().head(10)

#3.Count the top 10 directors with maximum movies
director_movie_count = df['director'].value_counts().dropna().head(10)

#Data Visualization

#1.Visualizing the Distribution of Movies and TV Shows 
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10,6))
sns.barplot(x='type',y='count',data=df1,palette='pastel',width=0.5)
plt.title('DISTRIBUTION OF MOVIES & TV SHOWS')
plt.xlabel('Genre')
plt.ylabel('Distirbution')
plt.show()

#2.Visualizing the Genre Distibution
plt.figure(figsize=(12,6))
sns.barplot(x=genre_counts.values, y=genre_counts.index,palette='Set2',hue=(genre_counts),width=0.8,dodge=False)
plt.title('Most common Genres on Netflix')
plt.xlabel('Distribution')
plt.ylabel('Genres')
plt.show()

#3.Visualizing the top 10 Directors with Maximum movies
plt.figure(figsize=(10, 6))
sns.barplot(x=director_movie_count.values, y=director_movie_count.index, palette='pastel',hue=(director_movie_count))
plt.title('Top 10 Directors with Highest Number of Movies on Netflix')
plt.xlabel('Number of Movies')
plt.ylabel('Director')
plt.show()

#4.Visualizing Movies and Tv Shows launched by Months

#Lets retrieve a list of months from the date_added column and count the number of movies and tv shows separately
month_added=df['date_added'].dt.month
df['month_added']=month_added #creating a df of the list of months

movie_counts=df[df['type']=='Movie']['month_added'].value_counts().sort_index()
tv_counts=df[df['type']=='TV Show']['month_added'].value_counts().sort_index()

plt.figure(figsize=(12, 9))
plt.plot(movie_counts.index,movie_counts.values,label='Movie')
plt.plot(tv_counts.index,tv_counts.values,label='TV Show')
plt.xlabel("Months")
plt.ylabel("Frequency of releases")
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.grid(True)
plt.suptitle("Monthly releases of Movies and TV shows on Netflix")
plt.show()

#5.Visualizing the count of number of Movie entries every year 

#Lets retrieve a list of years from the date_added column and count the number of years
year_added=df['date_added'].dt.year
df['year_added']=year_added
year_counts=df['year_added'].value_counts().sort_index()

plt.figure(figsize=(10, 8))
bars = plt.bar(year_counts.index, year_counts.values, color=sns.color_palette('pastel', n_colors=len(year_counts)))

# Set y-axis to logarithmic scale
plt.yscale('log')

plt.title('Count Of Movies Added By Year (Logarithmic Scale)', fontsize=16)
plt.xlabel('Year Added', fontsize=14)
plt.ylabel('Count (Log Scale)', fontsize=14)

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, int(height), ha='center', va='bottom', fontsize=12)

plt.show()

#6.Visualizing the frequency of common words used in movie titles using tag_cloud or wordcloud plot
movie_title = ' '.join(df['title'].dropna()) 
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(movie_title)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='spline16')
plt.axis('off')
plt.title('Word Cloud of Movie Titles on Netflix')
plt.show()

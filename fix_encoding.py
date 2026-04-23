# Fix for encoding issue
# Add this code in cell 21 AFTER encoding but BEFORE train_test_split

# Update X with the encoded data
X = data.drop('Churn', axis=1)
y = data['Churn']

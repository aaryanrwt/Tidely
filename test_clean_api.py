import tidely as td

result = td.clean("sample_test.csv")
print("df type:", type(result.df))
result.show()

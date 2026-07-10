from pipeline import Pipeline

def main():
    query = "Top 10 quant companies in india"
    
    pipeline = Pipeline()
    pipeline.run(query)


if __name__ == "__main__":
    main()
from pipeline import Pipeline

def main():
    query = "What are the main benefits of intermittent fasting?"
    
    pipeline = Pipeline()
    pipeline.run(query)


if __name__ == "__main__":
    main()
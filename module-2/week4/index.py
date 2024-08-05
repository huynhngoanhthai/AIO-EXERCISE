import main
import TEXT_RETRIEVAL
vi_data_df = TEXT_RETRIEVAL.create_data()


def question_1() -> None:
    x = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
    print(main.compute_mean(x))


def question_2() -> None:
    x = [1, 5, 4, 4, 9, 13]
    print(" Median : ", main.compute_median(x))


def question_3() -> None:
    x = [171, 176, 155, 167, 169, 182]
    print(main.compute_std(x))


def question_4() -> None:
    X = main.np.asarray([-2, -5, -11, 6, 4, 15, 9])
    Y = main.np.asarray([4, 25, 121, 36, 16, 225, 81])
    print(" Correlation : ", main.compute_correlation_coefficient(X, Y))


def question_5() -> None:
    data = main.get_data_csv()
    x = data['TV']
    y = data['Radio']
    corr_xy = main.compute_correlation_coefficient(x, y)
    print(f" Correlation between TV and Sales : { round (corr_xy , 2)}")


def question_6() -> None:
    data = main.get_data_csv()
    features = ['TV', 'Radio', 'Newspaper']

    for feature_1 in features:
        for feature_2 in features:
            correlation_value = main.compute_correlation_coefficient(
                data[feature_1], data[feature_2])

            print(
                f"Correlation between {feature_1} and {feature_2}: {round(correlation_value, 2)}")


def question_7() -> None:
    data = main.get_data_csv()
    x = data['Radio']
    y = data['Newspaper']

    result = main.np.corrcoef(x, y)

    print(result)


def question_8():
    data = main.get_data_csv()
    print(data.corr())
    return data.corr()


def question_9() -> None:

    main.visualize(question_8())


def question_10() -> None:
    print(TEXT_RETRIEVAL.process_text_data())


def question_11() -> None:
    context = vi_data_df['text']
    context = [doc.lower() for doc in context]
    tfidf_vectorizer = TEXT_RETRIEVAL.TfidfVectorizer()
    context_embedded = tfidf_vectorizer.fit_transform(context)

    question = vi_data_df.iloc[0]['question']
    results = TEXT_RETRIEVAL.tfidf_search(
        question, tfidf_vectorizer, context_embedded, top_d=5)
    print(results[0]['cosine_score'])


def question_12() -> None:
    context = vi_data_df['text']
    context = [doc.lower() for doc in context]
    tfidf_vectorizer = TEXT_RETRIEVAL.TfidfVectorizer()
    context_embedded = tfidf_vectorizer.fit_transform(context)

    question = vi_data_df.iloc[0]['question']
    results = TEXT_RETRIEVAL.corr_search(question, tfidf_vectorizer,
                                         context_embedded, top_d=5)
    print(results[1]['corr_score'])


if __name__ == "__main__":
    question_1()
    question_2()
    question_3()
    question_4()
    question_5()
    question_6()
    question_7()
    question_8()
    question_9()
    question_10()
    question_11()
    question_12()

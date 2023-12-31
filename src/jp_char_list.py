# あ行とか行の文字のみ学習
CHAR_LIST_A_to_KA = [
    "/177/", "/178/", "/179/", "/180/", "/181/",
    "/182/", "/183/", "/184/", "/185/", "/186/"
    ]

# さ行、た行、な行、は行
CHAR_LIST_SA_to_HA = [
    "/187/", "/188/", "/189/", "/190/", "/191/",
    "/192/", "/193/", "/194/", "/195/", "/196/",
    "/197/", "/198/", "/199/", "/200/", "/201/",
    "/202/", "/203/", "/204/", "/205/", "/206/"
    ]

# ひらがな全部 166, 177~221
CHAR_LIST_HIRAGANA = [
    "/166/"
]
for i in range(177, 222):
    temp = "/" + str(i) + "/"
    CHAR_LIST_HIRAGANA.append(temp)

# 祝を含む漢字20種類
CHAR_LIST_FOR_SYUKU = [
    "/15675/","/15677/","/15678/","/15685/","/15689/",
    "/15691/","/15696/","/15697/","/15698/","/15701/",
    "/15712/","/15715/","/15719/","/15720/","/15721/",
    "/15722/","/15723/","/15729/","/15732/","/15733/",
    ]
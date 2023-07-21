from dataclasses import dataclass

language_decode = {
    "default": "English",
    "ar": "Arabic",  # arabic
    "zh-tw": "Chinese Taiwan",  # chinese taiwan
    "zh-cn": "Chinese PRC",  # chinese PRC
    "zh-hk": "Chinese Hong Kong",  # chinese hong kong
    "zh-sg": "Chinese Singapore",  # chinese singapore,
    "fr": "French",  # french
    "de": "German",  # german
    "id": "Indonesian",  # indonesian
    "it": "Italian",  # italian
    "ja": "Japanese",  # japanese
    "ms": "Malaysian",  # malaysian
    "pt": "Portuguese",  # portuguese
    "es": "Spanish",  # spanish
    "tl": "Tagalog",  # tagalog
    "vi": "Vietnam",  # vietnam
}


@dataclass
class Locale:
    contact_us_today: str
    contact_info: str
    gallery: str
    have_a_question: str
    map: str


en = Locale(
    contact_us_today="Contact us today!",
    contact_info="Contact Info",
    gallery="Gallery",
    have_a_question="Have a Question?",
    map="Map"
)

ar = Locale(
    contact_us_today="اتصل بنا اليوم!",
    contact_info="معلومات الاتصال",
    gallery="معرض الصور",
    have_a_question="هل لديك سؤال؟",
    map="خريطة"
)

zh = Locale(
    contact_us_today="联系我们",
    contact_info="联系方式",
    gallery="图画",
    have_a_question="有疑问吗?",
    map="地图"
)

fr = Locale(
    contact_us_today="Contactez-nous aujourd'hui!",
    contact_info="Coordonnées de contact",
    gallery="Galerie",
    have_a_question="Vous avez une question?",
    map="Carte"
)

de = Locale(
    contact_us_today="Kontaktieren Sie uns noch heute!",
    contact_info="Kontaktinformationen",
    gallery="Galerie",
    have_a_question="Haben Sie eine Frage?",
    map="Karte"
)

id_ = Locale(
    contact_us_today="Hubungi kami hari ini!",
    contact_info="Informasi Kontak",
    gallery="Galeri",
    have_a_question="Ada pertanyaan?",
    map="Peta"
)

it = Locale(
    contact_us_today="Contattaci oggi stesso!",
    contact_info="Informazioni di contatto",
    gallery="Galleria",
    have_a_question="Hai una domanda?",
    map="Mappa"
)

ja = Locale(
    contact_us_today="今日お問い合わせください！",
    contact_info="連絡先情報",
    gallery="ギャラリー",
    have_a_question="質問はありますか？",
    map="地図"
)

ms = Locale(
    contact_us_today="Hubungi kami hari ini!",
    contact_info="Maklumat Hubungan",
    gallery="Galeri",
    have_a_question="Ada pertanyaan?",
    map="Peta"
)

pt = Locale(
    contact_us_today="Entre em contato conosco hoje!",
    contact_info="Informações de Contato",
    gallery="Galeria",
    have_a_question="Tem alguma pergunta?",
    map="Mapa"
)

es = Locale(
    contact_us_today="¡Contáctenos hoy!",
    contact_info="Información de contacto",
    gallery="Galería",
    have_a_question="¿Tienes alguna pregunta?",
    map="Mapa"
)

tl = Locale(
    contact_us_today="Makipag-ugnayan sa amin ngayon!",
    contact_info="Impormasyon sa Pakikipag-ugnayan",
    gallery="Galeriya",
    have_a_question="May tanong ka ba?",
    map="Mapa"
)

vi = Locale(
    contact_us_today="Liên hệ với chúng tôi ngay hôm nay!",
    contact_info="Thông tin liên hệ",
    gallery="Triển lãm ảnh",
    have_a_question="Có câu hỏi không?",
    map="Bản đồ"
)

language_locale = {
    "English": en,
    "Arabic": ar,
    "Chinese Taiwan": zh,
    "Chinese PRC": zh,
    "Chinese Hong Kong": zh,
    "Chinese Singapore": zh,
    "French": fr,
    "German": de,
    "Indonesian": id_,
    "Italian": it,
    "Japanese": ja,
    "Malaysian": ms,
    "Portuguese": pt,
    "Spanish": es,
    "Tagalog": tl,
    "Vietnam": vi
}


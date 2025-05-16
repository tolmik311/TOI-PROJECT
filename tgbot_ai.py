import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ConversationHandler,
    ContextTypes
)
import logging

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 1. Загрузка и подготовка данных
def load_and_prepare_data(filepath):
    df = pd.read_excel(filepath, sheet_name='Лист1')
    df['Outcome'] = df['Outcome'].apply(lambda x: 1 if x == 'Heart Attack' else 0)

    # Балансировка данных
    if df['Outcome'].mean() > 0.3:
        healthy_samples = df[df['Outcome'] == 0].sample(
            n=int(len(df) * 1.5),
            replace=True
        )
        df = pd.concat([df, healthy_samples], ignore_index=True)

    return df

# 2. Создание модели
def create_model():
    # Препроцессинг
    numeric_features = ['Age', 'Cholesterol', 'BloodPressure', 'HeartRate',
                      'PhysicalActivity', 'AlcoholConsumption', 'StressLevel']
    categorical_features = ['Gender', 'Diet']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

    # Модель с калибровкой вероятностей
    model = make_imb_pipeline(
        preprocessor,
        ADASYN(random_state=42),
        CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            ),
            method='isotonic',
            cv=5
        )
    )

    return model

# 3. Инициализация данных и модели
try:
    df = load_and_prepare_data('bd1.xlsx')
    model = create_model()
    model.fit(df.drop('Outcome', axis=1), df['Outcome'])
    logger.info("Модель успешно обучена")
except Exception as e:
    logger.error(f"Ошибка инициализации модели: {str(e)}")
    raise

# 4. Конфигурация бота
(
    START,
    AGE, GENDER, CHOLESTEROL, BLOOD_PRESSURE, HEART_RATE,
    SMOKER, DIABETES, HYPERTENSION, FAMILY_HISTORY,
    PHYSICAL_ACTIVITY, ALCOHOL_CONSUMPTION, DIET, STRESS_LEVEL
) = range(14)

gender_keyboard = [['Male', 'Female']]
diet_keyboard = [['Healthy', 'Moderate', 'Unhealthy']]
yes_no_keyboard = [['0 - Нет', '1 - Да']]

# 5. Обработчики диалога
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "🫀 Бот оценки риска сердечного приступа\n\n"
        "Для начала введите /predict\n"
        "Для отмены - /cancel"
    )
    return START

async def default_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("чтоб начать пользоваться ботом напишите /start")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()  # Очистка данных пользователя
    await update.message.reply_text("Введите ваш возраст (10-100 лет):")
    return AGE

async def age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        age = int(update.message.text)
        if not 10 <= age <= 100:
            await update.message.reply_text("Пожалуйста, введите возраст от 10 до 100:")
            return AGE
        context.user_data['Age'] = age
        reply_markup = ReplyKeyboardMarkup(gender_keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Ваш пол:Male(мужской), Female(женский)", reply_markup=reply_markup)
        return GENDER
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число:")
        return AGE

async def gender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    gender = update.message.text
    if gender not in ['Male', 'Female']:
        await update.message.reply_text(
            "Пожалуйста, выберите пол из предложенных:Male(мужской), Female(женский)",
            reply_markup=ReplyKeyboardMarkup(gender_keyboard, one_time_keyboard=True, resize_keyboard=True)
        )
        return GENDER
    context.user_data['Gender'] = gender
    await update.message.reply_text("Уровень холестерина (60-240 мг/дл):", reply_markup=ReplyKeyboardRemove())
    return CHOLESTEROL

async def cholesterol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        cholesterol = int(update.message.text)
        if not 60 <= cholesterol <= 240:
            await update.message.reply_text("Пожалуйста, введите значение от 60 до 240:")
            return CHOLESTEROL
        context.user_data['Cholesterol'] = cholesterol
        await update.message.reply_text("Артериальное давление (80-180 мм рт.ст.):")
        return BLOOD_PRESSURE
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число:")
        return CHOLESTEROL

async def blood_pressure(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        bp = int(update.message.text)
        if not 80 <= bp <= 180:
            await update.message.reply_text("Пожалуйста, введите значение от 80 до 180:")
            return BLOOD_PRESSURE
        context.user_data['BloodPressure'] = bp
        await update.message.reply_text("Пульс в покое (50-150 уд/мин):")
        return HEART_RATE
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число:")
        return BLOOD_PRESSURE

async def heart_rate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        hr = int(update.message.text)
        if not 50 <= hr <= 150:
            await update.message.reply_text("Пожалуйста, введите значение от 50 до 150:")
            return HEART_RATE
        context.user_data['HeartRate'] = hr
        reply_markup = ReplyKeyboardMarkup(yes_no_keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Курите ли вы?", reply_markup=reply_markup)
        return SMOKER
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число:")
        return HEART_RATE

async def smoker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        smoker = int(update.message.text.split()[0])
        if smoker not in [0, 1]:
            raise ValueError
        context.user_data['Smoker'] = smoker
        reply_markup = ReplyKeyboardMarkup(yes_no_keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Есть ли диабет?", reply_markup=reply_markup)
        return DIABETES
    except (ValueError, IndexError):
        reply_markup = ReplyKeyboardMarkup(yes_no_keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Пожалуйста, выберите вариант:", reply_markup=reply_markup)
        return SMOKER

async def diabetes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        diabetes = int(update.message.text.split()[0])
        if diabetes not in [0, 1]:
            raise ValueError
        context.user_data['Diabetes'] = diabetes
        reply_markup = ReplyKeyboardMarkup(yes_no_keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Есть ли гипертония?", reply_markup=reply_markup)
        return HYPERTENSION
    except (ValueError, IndexError):
        reply_markup = ReplyKeyboardMarkup(yes_no_keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Пожалуйста, выберите вариант:", reply_markup=reply_markup)
        return DIABETES

async def hypertension(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        hypertension = int(update.message.text.split()[0])
        if hypertension not in [0, 1]:
            raise ValueError
        context.user_data['Hypertension'] = hypertension
        reply_markup = ReplyKeyboardMarkup(yes_no_keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Были ли сердечные заболевания у родственников?", reply_markup=reply_markup)
        return FAMILY_HISTORY
    except (ValueError, IndexError):
        reply_markup = ReplyKeyboardMarkup(yes_no_keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Пожалуйста, выберите вариант:", reply_markup=reply_markup)
        return HYPERTENSION

async def family_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        family_history = int(update.message.text.split()[0])
        if family_history not in [0, 1]:
            raise ValueError
        context.user_data['FamilyHistory'] = family_history
        await update.message.reply_text("Уровень физической активности (1-10):", reply_markup=ReplyKeyboardRemove())
        return PHYSICAL_ACTIVITY
    except (ValueError, IndexError):
        reply_markup = ReplyKeyboardMarkup(yes_no_keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Пожалуйста, выберите вариант:", reply_markup=reply_markup)
        return FAMILY_HISTORY

async def physical_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        activity = int(update.message.text)
        if not 1 <= activity <= 10:
            await update.message.reply_text("Пожалуйста, введите число от 1 до 10:")
            return PHYSICAL_ACTIVITY
        context.user_data['PhysicalActivity'] = activity
        await update.message.reply_text("Употребление алкоголя (0-10):")
        return ALCOHOL_CONSUMPTION
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число от 1 до 10:")
        return PHYSICAL_ACTIVITY

async def alcohol_consumption(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        alcohol = int(update.message.text)
        if not 0 <= alcohol <= 10:
            await update.message.reply_text("Пожалуйста, введите число от 0 до 10:")
            return ALCOHOL_CONSUMPTION
        context.user_data['AlcoholConsumption'] = alcohol
        reply_markup = ReplyKeyboardMarkup(diet_keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.message.reply_text("Тип питания:Healthy(здоровое),Moderate(смешанное),Unhealthy(не здорове)", reply_markup=reply_markup)
        return DIET
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число от 0 до 10:")
        return ALCOHOL_CONSUMPTION

async def diet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    diet = update.message.text
    if diet not in ['Healthy', 'Moderate', 'Unhealthy']:
        await update.message.reply_text(
            "Пожалуйста, выберите тип питания:Healthy(здоровое),Moderate(смешанное),Unhealthy(не здорове)",
            reply_markup=ReplyKeyboardMarkup(diet_keyboard, one_time_keyboard=True, resize_keyboard=True)
        )
        return DIET
    context.user_data['Diet'] = diet
    await update.message.reply_text("Уровень стресса (0-10):", reply_markup=ReplyKeyboardRemove())
    return STRESS_LEVEL

async def stress_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        stress = int(update.message.text)
        if not 0 <= stress <= 10:
            await update.message.reply_text("Пожалуйста, введите число от 0 до 10:")
            return STRESS_LEVEL
        context.user_data['StressLevel'] = stress
        return await calculate_result(update, context)
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число от 0 до 10:")
        return STRESS_LEVEL

async def calculate_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        user_data = context.user_data

        # Подготовка данных для модели
        input_data = pd.DataFrame([{
            'Age': user_data['Age'],
            'Gender': user_data['Gender'],
            'Cholesterol': user_data['Cholesterol'],
            'BloodPressure': user_data['BloodPressure'],
            'HeartRate': user_data['HeartRate'],
            'Smoker': user_data['Smoker'],
            'Diabetes': user_data['Diabetes'],
            'Hypertension': user_data['Hypertension'],
            'FamilyHistory': user_data['FamilyHistory'],
            'PhysicalActivity': user_data['PhysicalActivity'],
            'AlcoholConsumption': user_data['AlcoholConsumption'],
            'Diet': user_data['Diet'],
            'StressLevel': user_data['StressLevel']
        }])

        # Предсказание
        proba = model.predict_proba(input_data)[0][1]
        risk_percent = proba * 100

        # Формирование ответа
        screenshot_url = "https://imgur.com/a/534CR8S"
        if risk_percent <20:
            message = f"✅ Очень низкий риск \n\nВаши показатели в отличном состоянии!\n\n📷 [для более подробного анализа посмотрите диаграммы ]({screenshot_url})"
        elif risk_percent < 25:
            message = f"🟢 Умеренный риск \n\nПродолжайте вести здоровый образ жизни.\n\n📷 [для более подробного анализа посмотрите диаграммы ]({screenshot_url})"
        elif risk_percent < 30:
            message = f"🟡 Высокий риск \n\nРекомендуется профилактический осмотр.\n\n📷 [для более подробного анализа посмотрите диаграммы ]({screenshot_url})"
        else:
            message = f"🔴 Очень высокий риск: \n\nРекомендуем обратиться к кардиологу.\n\n📷 [для более подробного анализа посмотрите диаграммы ]({screenshot_url})"

        await update.message.reply_text(
            f"🫀 Результат оценки риска:\n\n{message}\n\n"
            "Для новой оценки введите /predict",
            reply_markup=ReplyKeyboardRemove()
        )
    except Exception as e:
        logger.error(f"Ошибка расчета: {str(e)}")
        await update.message.reply_text(
            "Произошла ошибка при расчете. Пожалуйста, попробуйте позже.",
            reply_markup=ReplyKeyboardRemove()
        )

    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Оценка отменена.",
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

# 6. Запуск бота
def main():
    application = Application.builder().token("8090443890:AAHMrKakCXm8-0UXvskstcHGbX9C1z_9d5w").build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start),
                     CommandHandler('predict', predict)],
        states={
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, age)],
            GENDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, gender)],
            CHOLESTEROL: [MessageHandler(filters.TEXT & ~filters.COMMAND, cholesterol)],
            BLOOD_PRESSURE: [MessageHandler(filters.TEXT & ~filters.COMMAND, blood_pressure)],
            HEART_RATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, heart_rate)],
            SMOKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, smoker)],
            DIABETES: [MessageHandler(filters.TEXT & ~filters.COMMAND, diabetes)],
            HYPERTENSION: [MessageHandler(filters.TEXT & ~filters.COMMAND, hypertension)],
            FAMILY_HISTORY: [MessageHandler(filters.TEXT & ~filters.COMMAND, family_history)],
            PHYSICAL_ACTIVITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, physical_activity)],
            ALCOHOL_CONSUMPTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, alcohol_consumption)],
            DIET: [MessageHandler(filters.TEXT & ~filters.COMMAND, diet)],
            STRESS_LEVEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, stress_level)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, default_message))
    application.run_polling()

if __name__ == '__main__':
    main()

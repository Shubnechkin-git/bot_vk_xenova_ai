require('dotenv').config();
const { VK } = require('vk-io');
const TransformersApi = Function('return import("@xenova/transformers")')();

const vk = new VK({
    token: process.env.TOKEN,
});

const checkSubscribed = async (userId) => {
    return vk.api.groups.isMember({ "user_id": userId, "group_id": process.env.GROUP_ID })
        .then(isSubcribed => { return (isSubcribed == 1) })
        .catch(error => { return false })
}

const getUserInfo = async (userId) => {
    return vk.api.users.get({ "user_ids": [userId] })
        .then(userInfo => { return (userInfo) })
        .catch(error => { return error })
}

let pipe;
async function initializePipeline() {
    const { pipeline } = await TransformersApi;
    // pipe = await pipeline("text-generation", "Xenova/Qwen1.5-1.8B-Chat");
    pipe = await pipeline("text-generation", "Xenova/Qwen1.5-0.5B-Chat");
}

async function imageClassification(attachments) {
    const { pipeline } = await TransformersApi;
    const classifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224')
    let images = [];
    attachments.forEach(attachmentObject => {
        images.push(attachmentObject.largeSizeUrl);
    });

    let message = ""
    let result = await classifier(images).catch(error => {
        message = "не удалось определить. \n";
    });

    if (message) return message

    result.forEach(element => {
        message += element.label + "\n";
    });

    return message;
}

initializePipeline().then(() => {
    vk.updates.on('message', async (context, next) => {

        if (context.senderType !== 'user') {
            await next();
            return;
        }

        const isSubscribed = await checkSubscribed(context.senderId);
        const random_id = Math.floor(Math.random() * (1000 - 1 + 1)) + 1;
        const sendParametrs = { "user_id": context.senderId, "random_id": random_id };
        let messageText = "Вы не подписаны на наше сообщество. Пожалуйста, подпишитесь, для доступа к нейросети.";
        if (!isSubscribed) {
            context.send(messageText, sendParametrs).catch(error => console.log(error));
            await next();
            return;
        }

        if (!context.text && context.attachments.length == 0) {
            context.send("Я пока не готов ответить на это", sendParametrs).catch(error => console.log(error));
            await next();
            return;
        }

        messageText = "";
        if (context.attachments.length !== 0) {
            objectsInPhoto = (await imageClassification(context.attachments));
            messageText += "Возможно на фото: " + objectsInPhoto;
        }

        if (context.text) {
            const template = [
                { role: 'system', content: 'You are a helpful assistant.' },
                { role: 'user', content: context.text }
            ];
            const tokinezerOptions = { tokenize: false, add_generation_prompt: true };
            const text = await pipe.tokenizer.apply_chat_template(template, tokinezerOptions);
            const options = { max_new_tokens: 1000, do_sample: false, return_full_text: false };
            await pipe(text, options).then(result => messageText += result[0].generated_text);
        }

        context.send(messageText, sendParametrs).catch(error => console.log(error));

        await next();
    });

    vk.updates.start()
        .then(() => console.log('Бот запущен!'))
        .catch(console.error);
}); 

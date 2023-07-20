// Fetch the JSON file
window.onload = function() {
    fetch('data.json')
    .then(response => response.json())
    .then(data => {

        // Access the data in the JSON file
        let meta = data.meta;
        let banner = data.banner;
        let about = data.about;
        let blog = data.blogs;
        let gallery = data.gallery;
        let faq = data.faq;
        // ... and so on for all sections of your JSON ...

        // Update the HTML elements
        let metaTag = document.querySelector('meta[name="description"]');

        // If the meta tag doesn't exist yet, create it
        if (!metaTag) {
            metaTag = document.createElement('meta');
            metaTag.name = "description";
            document.head.appendChild(metaTag);
        }
        metaTag.content = meta.description;

        document.title = `${meta.title}`

        document.getElementById('banner').innerHTML = `
        <section style='
            background-image: url("data:image/jpg;base64,${banner.image}"); background-repeat: no-repeat; -webkit-background-size: cover;
            -moz-background-size: cover;
            -o-background-size: cover;
            background-size: cover;'>
            <div style='background-color: rgba(0,0,0, 0.5); margin: "auto"' class="banner__centered">
            <div class="banner__text">
                <div class="d-flex flex-column">
                <h1 class="heading-1 text-center">${banner.h1}</h1>
                <h2 class="heading-2 text-center">${banner.h2}</h2>
                <div class="mx-auto">
                    <button class="button1" href="#aboutus">${banner.button[0]}</button>
                    <button class="button1">${banner.button[1]}</button>
                </div>
                </div>
            </div>
            </div>
        </section>
        `;
        

        document.getElementById('aboutus').innerHTML = `
        <section class="bg-lightgray">
            <div class="banner__sidebyside">
                <div class="banner__text">
                    <div  data-aos="zoom-in-up" data-aos-anchor-placement="top-center">
                        <h1 class="banner__heading heading-1">${about.h2}</h1>
                        <h5 class="banner_subheading heading-5">${about.p}</h5>
                        <button class="learn-more">About Us</button>
                    </div>
                </div>
                <div class="banner__image" data-aos="flip-right" data-aos-anchor-placement="top-center">
                    <div class="img-wrapper">
                        <img src="data:image/jpeg;base64,${about.image}" alt="" />
                    </div>
                </div>
            </div>
            <!-- <div class="toolbar">Toolbar</div> -->
        </section>`;


        // document.getElementById('about').innerHTML = `<h2>${about.h2}</h2><p>${about.p}</p>`;
        // ... and so on for all sections of your JSON ...

        document.getElementById('blogtitle').innerHTML = `${blog.h2}`;

        blog.post.forEach(post => {
            document.getElementById('blogpost').innerHTML += `
            <div class="cell">
                <div class="first-content">
                    <h5 class="heading-5">${post.h3}</h5>
                </div>
                <div class="second-content">
                    <p>
                                ${post.p}
                    </p>
                </div>
            </div>`;
        });

        document.getElementById('galleryheader').innerHTML = `${gallery.h2}`;

        gallery.image.forEach(img => {
            document.getElementById('gallery').innerHTML += `
            <img src="data:image/jpeg;base64,${img}" alt="image.alt"/>
            `;
        });

        document.getElementById('faqtitle').innerHTML = `${faq.h2}`;

        faq.question.forEach(q => {
            document.getElementById('faq').innerHTML += `
            <input id="'collapsible${q.id}" class="toggle" type="checkbox"/>
            <label for="'collapsible${q.id}" class="lbl-toggle">${q.h3}</label>
            <div class="collapsible-content">
              <div class="content-inner">
                <p>
                ${q.p}
                </p>
              </div>
            </div>`;
        });

    }).catch((error) => {
        console.error('Error:', error);
    });
}
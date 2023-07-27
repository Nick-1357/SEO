// Fetch the JSON file
window.onload = function() {
    fetch('../data.json')
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
        <h1>
            ${banner.h1}
            
        </h1>
        <p>
        ${banner.h2}
        </p>
        <div class="btn-box">
            <a href="" class="btn-1">
            ${banner.button[0]}
            </a>
            <a href="" class="btn-2">
            ${banner.button[1]}
            </a>
        </div>
        `;

        document.getElementById('bannerimage').innerHTML = `
        <img src="data:image/jpg;base64,${banner.image}" alt="" />
        `;

        document.getElementById("aboutimage").innerHTML = `
        <img src="data:image/jpeg;base64,${about.image}" alt="" />      
        `;
        
        document.getElementById('about').innerHTML = `
        <div class="heading_container">
              <h2>
                ${about.h2}
              </h2>
            </div>
            <p>
              ${about.p}
            </p>
            <a href="">
              Read More
            </a>
        `;
        
        // document.getElementById('galleryheader').innerHTML = `${gallery.h2}`;

        // document.getElementById('gallery').innerHTML = `
        // <div class="box-1">
        //   <div class="img-box b-1">
        //   <img src="data:image/jpeg;base64,${gallery.image[0]}" alt="image.alt"/>
        //     <div class="btn-box">
        //       <a href="" class="btn-1">
        //         <i class="fa fa-share-alt" aria-hidden="true"></i>
        //       </a>
        //     </div>
        //   </div>
        //   <div class="img-box b-2">
        //   <img src="data:image/jpeg;base64,${gallery.image[1]}" alt="image.alt"/>
        //     <div class="btn-box">
        //       <a href="" class="btn-1">
        //         <i class="fa fa-share-alt" aria-hidden="true"></i>
        //       </a>
        //     </div>
        //   </div>
        // </div>
        // <div class="box-2">
        //   <div class="box-2-top">
        //     <div class="img-box b-3">
        //     <img src="data:image/jpeg;base64,${gallery.image[2]}" alt="image.alt"/>
        //       <div class="btn-box">
        //         <a href="" class="btn-1">
        //           <i class="fa fa-share-alt" aria-hidden="true"></i>
        //         </a>
        //       </div>
        //     </div>
        //     <div class="img-box b-2">
        //     <img src="data:image/jpeg;base64,${gallery.image[3]}" alt="image.alt"/>
        //     <div class="btn-box">
        //       <a href="" class="btn-1">
        //         <i class="fa fa-share-alt" aria-hidden="true"></i>
        //       </a>
        //     </div>
        //   </div>
        //   </div>
        // </div>
        // <div class="box-3">
        //   <div class="img-box b-1">
        //   <img src="data:image/jpeg;base64,${gallery.image[4]}" alt="image.alt"/>
        //     <div class="btn-box">
        //       <a href="" class="btn-1">
        //         <i class="fa fa-share-alt" aria-hidden="true"></i>
        //       </a>
        //     </div>
        //   </div>
        //   <div class="img-box b-2">
        //   <img src="data:image/jpeg;base64,${gallery.image[5]}" alt="image.alt"/>
        //     <div class="btn-box">
        //       <a href="" class="btn-1">
        //         <i class="fa fa-share-alt" aria-hidden="true"></i>
        //       </a>
        //     </div>
        //   </div>
        // </div>
        // `;



        // // document.getElementById('about').innerHTML = `<h2>${about.h2}</h2><p>${about.p}</p>`;
        // // ... and so on for all sections of your JSON ...

        document.getElementById('blogtitle').innerHTML = `<h2>${blog.h2}</h2>`;
        // document.getElementById('blogt').innerHTML = `${blog.h2}`;
        // document.getElementById('blogt2').innerHTML = `${blog.h2}`;

        document.getElementById('blogpost').innerHTML += `    
          <div class="carousel-item active">
            <div class="container">
              <div class="row">
                <div class="col-md-10 mx-auto">
                  <div class="box">
                    <div class="detail-box">
                      <div class="client_info">
                        <div class="client_name">
                          <h5>
                            ${blog.post[0].h3}
                          </h5>
                        </div>
                        <i class="fa fa-quote-left" aria-hidden="true"></i>
                      </div>
                      <p>
                        ${blog.post[0].p}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="carousel-item">
            <div class="container">
              <div class="row">
                <div class="col-md-10 mx-auto">
                  <div class="box">
                    <div class="detail-box">
                      <div class="client_info">
                        <div class="client_name">
                          <h5>
                            ${blog.post[1].h3}
                          </h5>
                        </div>
                        <i class="fa fa-quote-left" aria-hidden="true"></i>
                      </div>
                      <p>
                        ${blog.post[1].p}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="carousel-item">
            <div class="container">
              <div class="row">
                <div class="col-md-10 mx-auto">
                  <div class="box">
                    <div class="detail-box">
                      <div class="client_info">
                        <div class="client_name">
                          <h5>
                            ${blog.post[2].h3}
                          </h5>
                        </div>
                        <i class="fa fa-quote-left" aria-hidden="true"></i>
                      </div>
                      <p>
                        ${blog.post[2].p}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

            `;
            

        // // gallery.image.forEach(img => {
        // //     document.getElementById('gallery').innerHTML += `
        // //     <img src="data:image/jpeg;base64,${img}" alt="image.alt"/>
        // //     `;
        // // });

        // // document.getElementById('aboutus').innerHTML = `
        // // <section class="bg-lightgray">
        // //     <div class="banner__sidebyside">
        // //         <div class="banner__text">
        // //             <div  data-aos="zoom-in-up" data-aos-anchor-placement="top-center">
        // //                 <h1 class="banner__heading heading-1">${about.h2}</h1>
        // //                 <h5 class="banner_subheading heading-5">${about.p}</h5>
        // //                 <button class="learn-more">About Us</button>
        // //             </div>
        // //         </div>
        // //         <div class="banner__image" data-aos="flip-right" data-aos-anchor-placement="top-center">
        // //             <div class="img-wrapper">
        // //                 <img src="data:image/jpeg;base64,${about.image}" alt="" />
        // //             </div>
        // //         </div>
        // //     </div>
        // //     <!-- <div class="toolbar">Toolbar</div> -->
        // // </section>`;



        // // document.getElementById('about').innerHTML = `<h2>${about.h2}</h2><p>${about.p}</p>`;
        // // ... and so on for all sections of your JSON ...

        // // document.getElementById('blogtitle').innerHTML = `${blog.h2}`;

        // // blog.post.forEach(post => {
        // //     document.getElementById('blogpost').innerHTML += `
        // //     <div class="cell">
        // //         <div class="first-content">
        // //             <h5 class="heading-5">${post.h3}</h5>
        // //         </div>
        // //         <div class="second-content">
        // //             <p>
        // //                         ${post.p}
        // //             </p>
        // //         </div>
        // //     </div>`;
        // // });

        // // document.getElementById('galleryheader').innerHTML = `${gallery.h2}`;

        gallery.image.forEach(img => {
            document.getElementById('gallery').innerHTML += `
            <div class="col-md-6 col-lg-4">
              <div class="box ">
                <div class="img-box">
                <img src="data:image/jpeg;base64,${img}" alt="image.alt"/>
                </div>
              </div>
            </div>
            `;
        });

        // document.getElementById('faqtitle').innerHTML = `${faq.h2}`;

        // faq.question.forEach(q => {
        //     document.getElementById('faq').innerHTML += `            
        //     <div class="faq_item">
        //       <h3 class="faq_question" onclick="toggleAnswer('faq${q.id}')">
        //         ${q.h3}
        //       </h3>
        //       <p class="faq_answer" id="faq${q.id}" style="display:none;">
        //         ${q.p}
        //       </p>
        //     `;
        // });

    }).catch((error) => {
        console.error('Error:', error);
    });
}
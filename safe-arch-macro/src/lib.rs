// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use proc_macro::TokenStream;
use syn::{parse::Parser, FnArg, GenericParam};

#[cfg(feature = "nightly")]
#[proc_macro_attribute]
pub fn safe_arch(_: TokenStream, input: TokenStream) -> TokenStream {
    input
}

#[cfg(not(feature = "nightly"))]
#[proc_macro_attribute]
pub fn safe_arch(_: TokenStream, input: TokenStream) -> TokenStream {
    use quote::quote;
    use syn::{parse_macro_input, Item, Signature};
    let Item::Fn(func) = parse_macro_input!(input as Item) else {
        panic!("safe_arch not applied to a function");
    };
    let attrs = &func.attrs;
    let vis = &func.vis;
    let block = &func.block;
    let Signature {
        constness,
        asyncness,
        unsafety,
        abi,
        fn_token,
        ident,
        generics,
        paren_token: _paren_token,
        inputs,
        variadic,
        output,
    } = &func.sig;
    let (impl_generics, _, where_clause) = generics.split_for_impl();

    if unsafety.is_some() {
        panic!("safe_arch applied to already-unsafe function");
    }

    quote! {
        #(#attrs)* #vis #constness #asyncness unsafe #abi
        #fn_token #ident #impl_generics (#inputs #variadic) #output #where_clause {
            #block
        }
    }
    .into()
}

#[proc_macro_attribute]
pub fn safe_arch_entrypoint(args: TokenStream, input: TokenStream) -> TokenStream {
    use quote::quote;
    use syn::{parse_macro_input, Item, Signature};

    let target_features =
        syn::punctuated::Punctuated::<syn::LitStr, syn::Token![,]>::parse_terminated
            .parse(args)
            .unwrap();

    let tf = target_features
        .iter()
        .map(|x| x.value())
        .collect::<Vec<_>>()
        .join(",");

    let f_checks = target_features
        .iter()
        .map(|x| {
            quote! { is_x86_feature_detected!(#x) }
        })
        .collect::<Vec<_>>();

    let Item::Fn(func) = parse_macro_input!(input as Item) else {
        panic!("safe_arch_entrypoint not applied to a function");
    };
    let attrs = &func.attrs;
    let vis = &func.vis;
    let block = &func.block;
    let Signature {
        constness,
        asyncness,
        unsafety,
        abi,
        fn_token,
        ident,
        generics,
        paren_token: _paren_token,
        inputs,
        variadic,
        output,
    } = &func.sig;
    let generics_names = generics
        .params
        .iter()
        .filter_map(|x| match x {
            GenericParam::Type(t) => Some(t.ident.clone()),
            GenericParam::Const(c) => Some(c.ident.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let inner_generics = if generics_names.is_empty() {
        quote! {}
    } else {
        quote! { :: < #(#generics_names),* > }
    };
    let (impl_generics, _, where_clause) = generics.split_for_impl();

    if unsafety.is_some() {
        panic!("safe_arch_entrypoint applied to already-unsafe function");
    }

    let arg_names = inputs
        .iter()
        .map(|x| match x {
            FnArg::Typed(pat) => {
                let pat = &pat.pat;
                quote! { #pat }
            }
            FnArg::Receiver(r) => {
                quote! { #r }
            }
        })
        .collect::<Vec<_>>();

    #[cfg(not(feature = "nightly"))]
    let inner_unsafety = quote! { unsafe };

    #[cfg(feature = "nightly")]
    let inner_unsafety = quote! {};

    quote! {
        #(#attrs)* #vis #constness #asyncness #abi
        #fn_token #ident #impl_generics (#inputs #variadic) #output #where_clause {
            #[target_feature(enable = #tf)]
            #inner_unsafety fn inner_fn #impl_generics (#inputs #variadic) #output #where_clause {
                #block
            }
            if #(#f_checks &&)* true {
                // SAFETY: target features are checked by the if condition.
                unsafe {
                    inner_fn #inner_generics (#(#arg_names,)*)
                }
            } else {
                panic!("Your CPU does not support some required target features");
            }
        }
    }
    .into()
}
